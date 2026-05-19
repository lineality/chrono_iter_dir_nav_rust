// src/main.rs
//
// Demo: chrono_sort_hash_to_n and check_chronosort_hash_to_n
//
// These two functions implement the Plan-B sequence-integrity check
// described in the project discussion. The core question they answer is:
//
//   "Has the chronological ordering of files I have already processed
//    been retroactively changed by a delayed-thread mtime insertion?"
//
// This is distinct from "have new files appeared?" (which update_index
// already handles). It detects the chess-game edge case where a slow
// thread finishes writing a move-file whose mtime is earlier than moves
// the engine has already read, silently shifting earlier positions.
//
// Demo layout:
//   test/
//     watched_hash_demo/   Watched directory (created if absent).
//                          Files from prior runs accumulate here by
//                          design; each run adds its own timestamped
//                          batch. The cleanup prompt removes only the
//                          files created by THIS run.
//     temp_hash_demo/      Index temp root (created if absent).
//
// Run:   cargo run

use std::io::Write;
use std::path::{Path, PathBuf};

mod chrono_sort_module;

use chrono_sort_module::{
    MAX_FULL_PATH_LEN, UpdateOutcome, check_chronosort_hash_to_n, chrono_sort_hash_to_n,
    count_committed_files, lookup_abs_file_path_at_mtime_chronological_index, purge_index_state,
    update_index,
};

// =========================================================================
// Demo helpers
// =========================================================================

/// Resolves `<cargo_manifest_dir>/test/<subdir>`, creating the directory
/// if absent. Demo-only; not used by the index module itself.
fn resolve_demo_subdir(relative_subdir: &str) -> Result<PathBuf, std::io::Error> {
    let crate_root = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let mut full = PathBuf::from(crate_root);
    full.push("test");
    full.push(relative_subdir);
    std::fs::create_dir_all(&full)?;
    Ok(full)
}

/// Returns a nanosecond-precision timestamp string suitable for use in
/// filenames: `<seconds>_<nanos>`. Both components are zero-padded to
/// fixed widths so lexicographic order matches chronological order.
///
/// Using a timestamp prefix per demo run means each run creates files
/// with unique basenames, so files from prior runs do not collide with
/// files from this run. The index accumulates all runs' files, which
/// correctly demonstrates the "growing directory" steady-state.
fn timestamp_prefix() -> String {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(std::time::Duration::ZERO);
    // Zero-pad seconds to 10 digits (sufficient past year 2286) and
    // nanos to 9 digits.
    format!("{:010}_{:09}", duration.as_secs(), duration.subsec_nanos())
}

/// Creates a file whose basename is `<run_prefix>_<label>.pgn` in
/// `watched_dir`. Sleeps 15 ms first so each call produces a file with
/// a strictly newer mtime than any file created by the previous call
/// (on filesystems with millisecond-resolution timestamps).
///
/// Returns `(created_path, basename_string)` so the caller can track
/// which files this run created (for the cleanup prompt).
fn create_watched_file(
    watched_dir: &Path,
    run_prefix: &str,
    label: &str,
) -> Result<(PathBuf, String), std::io::Error> {
    std::thread::sleep(std::time::Duration::from_millis(15));
    let basename = format!("{}_{}.pgn", run_prefix, label);
    let mut path = PathBuf::from(watched_dir);
    path.push(&basename);
    let mut handle = std::fs::File::create(&path)?;
    // Content is the label itself — distinguishable in hex dumps but
    // not meaningful to the index (which only reads mtimes + basenames).
    handle.write_all(label.as_bytes())?;
    handle.sync_all()?;
    Ok((path, basename))
}

/// Prints a one-line label for an UpdateOutcome.
fn outcome_label(outcome: UpdateOutcome) -> &'static str {
    match outcome {
        UpdateOutcome::ColdBuildCompleted => "ColdBuildCompleted",
        UpdateOutcome::RebuiltDueToInconsistency => "RebuiltDueToInconsistency",
        UpdateOutcome::NoChangesDetected => "NoChangesDetected",
        UpdateOutcome::IncrementalAppendCompleted => "IncrementalAppendCompleted",
    }
}

// =========================================================================
// Cleanup prompt
// =========================================================================

/// Asks the user whether to clean up the demo's index state and ALL
/// files in the watched demo directory.
///
/// The watched directory (`watched_hash_demo/`) is a demo-only scratch
/// space. When the user confirms cleanup, every regular file inside it
/// is removed — not just the files created by this run. This clears
/// accumulated files from all prior runs.
///
/// Files created by this run are listed before the prompt so the user
/// can see what exists. The total file count in the watched dir is also
/// shown so prior-run accumulation is visible.
///
/// Cleanup steps when the user answers yes:
///   1. `purge_index_state` — removes `<temp_root>/chrono_index/` and
///      all index files inside it.
///   2. Remove every regular file inside `watched_dir` (all runs).
///   3. Attempt (non-destructively) to remove the now-empty
///      `watched_dir` and `temp_root` demo subdirectories.
///
/// Per project policy: no panic, no halt. Every step that fails is
/// reported with a terse message and skipped.
fn prompt_cleanup(temp_root: &Path, watched_dir: &Path, this_run_basenames: &[String]) {
    use std::io::{BufRead, Write};

    // Count total files currently in the watched dir so the user can
    // see how many runs have accumulated.
    let total_in_watched = count_regular_files_in_dir(watched_dir);

    println!();
    println!("Files created this run ({}):", this_run_basenames.len());
    for name in this_run_basenames {
        println!("  {}", name);
    }
    if total_in_watched > this_run_basenames.len() as u64 {
        println!(
            "  ... plus {} file(s) from prior runs (will also be removed on yes)",
            total_in_watched.saturating_sub(this_run_basenames.len() as u64)
        );
    }
    println!();
    print!(
        "Clean up ALL demo files ({} total) and chrono-sort index? [y/N] ",
        total_in_watched
    );

    if std::io::stdout().flush().is_err() {
        // Cannot flush prompt; safe default is keep files.
        return;
    }

    let stdin_handle = std::io::stdin();
    let mut response_line = String::new();
    match stdin_handle.lock().read_line(&mut response_line) {
        Ok(0) => {
            println!("(no input — keeping files)");
            return;
        }
        Ok(_) => {}
        Err(_) => {
            println!("(could not read response — keeping files)");
            return;
        }
    }

    let trimmed = response_line.trim();
    let user_confirmed = matches!(trimmed, "y" | "Y" | "yes" | "Yes" | "YES");

    if !user_confirmed {
        println!("keeping files");
        return;
    }

    println!();
    println!("--- cleanup ---");

    // Step 1: purge the chrono-sort index state.
    // purge_index_state removes <temp_root>/chrono_index/ and its
    // contents. It does not touch temp_root itself.
    match purge_index_state(temp_root) {
        Ok(()) => println!("  purge_index_state: ok"),
        Err(e) => println!("  purge_index_state: {} (continuing)", e.code()),
    }

    // Step 2: remove ALL regular files in the watched directory.
    // This is a demo-only directory; the user confirmed removing
    // everything in it. Subdirectories (if any) are left alone.
    let mut files_removed: u32 = 0;
    let mut files_failed: u32 = 0;

    match std::fs::read_dir(watched_dir) {
        Err(_) => {
            println!("  watched dir read error — skipping file removal");
        }
        Ok(dir_iter) => {
            for entry_result in dir_iter {
                let entry = match entry_result {
                    Ok(e) => e,
                    Err(_) => {
                        files_failed = files_failed.saturating_add(1);
                        continue;
                    }
                };

                // Only remove regular files; leave any subdirectories.
                let file_type = match entry.file_type() {
                    Ok(ft) => ft,
                    Err(_) => {
                        files_failed = files_failed.saturating_add(1);
                        continue;
                    }
                };
                if !file_type.is_file() {
                    continue;
                }

                match std::fs::remove_file(entry.path()) {
                    Ok(()) => {
                        files_removed = files_removed.saturating_add(1);
                    }
                    Err(e) => {
                        // Report per-file failures but continue with
                        // the rest. Never halt.
                        println!(
                            "  could not remove {}: {} (continuing)",
                            entry.file_name().to_string_lossy(),
                            e.kind()
                        );
                        files_failed = files_failed.saturating_add(1);
                    }
                }
            }
        }
    }

    println!(
        "  watched dir: {} file(s) removed, {} issues",
        files_removed, files_failed
    );

    // Step 3: attempt to remove the demo subdirectories only if now
    // empty. remove_dir (not remove_dir_all) succeeds only on an
    // empty directory — it cannot accidentally remove unexpected files.
    for (label, dir) in &[("watched_dir", watched_dir), ("temp_root", temp_root)] {
        match std::fs::remove_dir(dir) {
            Ok(()) => println!("  removed {} (now empty): {}", label, dir.display()),
            Err(e) => println!("  {} kept ({}): {}", label, e.kind(), dir.display()),
        }
    }

    println!("--- cleanup done ---");
}

/// Counts the number of regular files (not subdirectories) directly
/// inside `dir`. Returns 0 on any error. Used only to show the user
/// the accumulated file count before the cleanup prompt.
fn count_regular_files_in_dir(dir: &Path) -> u64 {
    let dir_iter = match std::fs::read_dir(dir) {
        Ok(it) => it,
        Err(_) => return 0,
    };
    let mut count: u64 = 0;
    for entry_result in dir_iter {
        let entry = match entry_result {
            Ok(e) => e,
            Err(_) => continue,
        };
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if file_type.is_file() {
            count = count.saturating_add(1);
        }
    }
    count
}

// =========================================================================
// Main
// =========================================================================

fn main() {
    println!("=== chrono_sort_hash_to_n / check_chronosort_hash_to_n demo ===\n");

    // --- Step 0: resolve demo directories --------------------------------
    let watched_dir = match resolve_demo_subdir("watched_hash_demo") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("setup: cannot create watched dir: {}", e);
            return;
        }
    };
    let temp_root = match resolve_demo_subdir("temp_hash_demo") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("setup: cannot create temp dir: {}", e);
            return;
        }
    };
    println!("watched dir : {}", watched_dir.display());
    println!("temp root   : {}", temp_root.display());

    // One timestamp prefix shared by all files created in this run.
    // Format: <secs>_<nanos>  e.g. 1779200900_123456789
    // This guarantees basenames are unique across runs and that files
    // from this run can be precisely identified at cleanup time.
    let run_prefix = timestamp_prefix();
    println!("run prefix  : {}", run_prefix);
    println!();

    // Accumulate the basenames of every file this run creates so the
    // cleanup prompt can remove exactly those files and nothing else.
    let mut this_run_basenames: Vec<String> = Vec::new();

    // =========================================================================
    // Phase 1: Build initial index with three files.
    //
    // Because the watched directory may already contain files from prior
    // runs, update_index may return ColdBuildCompleted (first ever run),
    // NoChangesDetected (if this run's files happen to match the index
    // exactly — impossible with unique prefixes), or
    // IncrementalAppendCompleted (the normal case when prior files are
    // already indexed and three new ones just appeared).
    //
    // In all cases `count_committed_files` returns the true total, which
    // is what Phase 1's hash is computed over.
    // =========================================================================

    println!("--- Phase 1: create three move files and build/update index ---");

    for label in &["move_001_white", "move_002_black", "move_003_white"] {
        match create_watched_file(&watched_dir, &run_prefix, label) {
            Ok((path, basename)) => {
                println!("  created: {}", path.display());
                this_run_basenames.push(basename);
            }
            Err(e) => {
                eprintln!("  could not create {}: {}", label, e);
                return;
            }
        }
    }

    let summary = match update_index(&temp_root, &watched_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("update_index failed: {}", e.code());
            return;
        }
    };
    println!(
        "  update_index: outcome={} file_count={}",
        outcome_label(summary.outcome),
        summary.final_file_count
    );

    let total_after_phase1 = match count_committed_files(&temp_root) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("count_committed_files failed: {}", e.code());
            return;
        }
    };
    if total_after_phase1 == 0 {
        eprintln!("index is empty after phase 1; aborting demo");
        return;
    }

    // Record the hash of the full committed sequence (all files, not
    // just this run's three). This is what the engine would store.
    let last_position_phase1 = total_after_phase1.saturating_sub(1);
    let stored_hash = match chrono_sort_hash_to_n(&temp_root, last_position_phase1) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("chrono_sort_hash_to_n failed: {}", e.code());
            return;
        }
    };
    println!(
        "\n  chrono_sort_hash_to_n(position={}) = 0x{:016X}",
        last_position_phase1, stored_hash
    );
    println!(
        "  (engine stores this hash after processing all {} committed files)",
        total_after_phase1
    );

    // Show the current chronological order for reference.
    println!("\n  Current chronological order (all committed files):");
    let mut path_buf = [0u8; MAX_FULL_PATH_LEN];
    for position in 0..total_after_phase1 {
        match lookup_abs_file_path_at_mtime_chronological_index(&temp_root, position, &mut path_buf)
        {
            Ok(Some(result)) => {
                let display = String::from_utf8_lossy(&path_buf[..result.path_byte_length]);
                println!(
                    "    [{}] mtime={}.{:06} {}",
                    position,
                    result.looked_up_file_mtime_sec,
                    result.looked_up_file_mtime_nsec / 1_000,
                    display
                );
            }
            Ok(None) => println!("    [{}] (no file)", position),
            Err(e) => println!("    [{}] lookup error: {}", position, e.code()),
        }
    }

    // =========================================================================
    // Phase 2: Periodic poll — no new files.
    // Expected: NoChangesDetected + check returns Ok(true).
    // =========================================================================

    println!("\n--- Phase 2: periodic poll — no new files ---");

    let poll_summary = match update_index(&temp_root, &watched_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("update_index (poll) failed: {}", e.code());
            return;
        }
    };
    println!(
        "  update_index: outcome={}",
        outcome_label(poll_summary.outcome)
    );

    match check_chronosort_hash_to_n(&temp_root, last_position_phase1, stored_hash) {
        Ok(true) => println!(
            "  check_chronosort_hash_to_n(position={}, hash=0x{:016X})\n  \
             -> Ok(true)  past sequence INTACT — no rebuild needed",
            last_position_phase1, stored_hash
        ),
        Ok(false) => println!(
            "  check_chronosort_hash_to_n\n  \
             -> Ok(false) past sequence CHANGED — engine would rebuild state"
        ),
        Err(e) => eprintln!("  check_chronosort_hash_to_n error: {}", e.code()),
    }

    // =========================================================================
    // Phase 3: Append one new file.
    // Expected: IncrementalAppendCompleted + prefix check returns Ok(true).
    // =========================================================================

    println!("\n--- Phase 3: append one new move file ---");

    match create_watched_file(&watched_dir, &run_prefix, "move_004_black") {
        Ok((path, basename)) => {
            println!("  created: {}", path.display());
            this_run_basenames.push(basename);
        }
        Err(e) => {
            eprintln!("  could not create move_004: {}", e);
            return;
        }
    }

    let append_summary = match update_index(&temp_root, &watched_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("update_index (append) failed: {}", e.code());
            return;
        }
    };
    println!(
        "  update_index: outcome={} file_count={}",
        outcome_label(append_summary.outcome),
        append_summary.final_file_count
    );

    // The prefix (positions 0..=last_position_phase1) must be unchanged.
    match check_chronosort_hash_to_n(&temp_root, last_position_phase1, stored_hash) {
        Ok(true) => println!(
            "  check_chronosort_hash_to_n(position={}, hash=0x{:016X})\n  \
             -> Ok(true)  prefix INTACT — engine reads only new move at position {}",
            last_position_phase1,
            stored_hash,
            append_summary.final_file_count.saturating_sub(1)
        ),
        Ok(false) => println!(
            "  check_chronosort_hash_to_n\n  \
             -> Ok(false) prefix CHANGED — engine would rebuild all state from position 0"
        ),
        Err(e) => eprintln!("  check_chronosort_hash_to_n error: {}", e.code()),
    }

    // Update the stored hash to cover all four of this run's files
    // (plus any prior-run files already in the index).
    let total_after_phase3 = match count_committed_files(&temp_root) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("count_committed_files failed: {}", e.code());
            return;
        }
    };
    let last_position_phase3 = total_after_phase3.saturating_sub(1);
    let stored_hash_phase3 = match chrono_sort_hash_to_n(&temp_root, last_position_phase3) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("chrono_sort_hash_to_n (phase 3) failed: {}", e.code());
            return;
        }
    };
    println!(
        "  chrono_sort_hash_to_n(position={}) = 0x{:016X}  (updated stored hash)",
        last_position_phase3, stored_hash_phase3
    );

    // =========================================================================
    // Phase 4: Demonstrate Ok(false) — stale hash passed to check.
    //
    // stored_hash was computed over positions 0..=last_position_phase1.
    // We now ask check to cover positions 0..=last_position_phase3
    // (one position further). The sequences differ in length so their
    // hashes differ, and check returns false.
    //
    // This is the detection path: the engine's stored hash from before
    // the append does not match the longer current sequence.
    // =========================================================================

    println!("\n--- Phase 4: demonstrate Ok(false) with a stale stored hash ---");

    match check_chronosort_hash_to_n(&temp_root, last_position_phase3, stored_hash) {
        Ok(true) => println!(
            "  check(position={}, stale_hash=0x{:016X})\n  \
             -> Ok(true)  (unexpected: hash collision or sequences happen to match)",
            last_position_phase3, stored_hash
        ),
        Ok(false) => println!(
            "  check(position={}, stale_hash=0x{:016X})\n  \
             -> Ok(false) CHANGE DETECTED — stale hash does not cover the new position\n  \
             engine discards state and rebuilds from position 0",
            last_position_phase3, stored_hash
        ),
        Err(e) => eprintln!("  check error: {}", e.code()),
    }

    // =========================================================================
    // Phase 5: Demonstrate Err — position past end of index.
    // =========================================================================

    println!("\n--- Phase 5: demonstrate Err for out-of-range position ---");

    let out_of_range = total_after_phase3.saturating_add(99);
    match check_chronosort_hash_to_n(&temp_root, out_of_range, stored_hash_phase3) {
        Ok(v) => println!(
            "  check(position={}) -> Ok({}) (unexpected)",
            out_of_range, v
        ),
        Err(e) => println!(
            "  check(position={}) -> Err({}) \
             out-of-range correctly returns error — engine rebuilds defensively",
            out_of_range,
            e.code()
        ),
    }

    println!("\n=== demo complete ===");
    println!();
    println!("Summary of Plan-B usage pattern: (not results of demo)");
    println!();
    println!("  after update_index commits K files:");
    println!("    stored_hash = chrono_sort_hash_to_n(temp_root, K-1)");
    println!();
    println!("  on each subsequent poll:");
    println!("    match check_chronosort_hash_to_n(temp_root, K-1, stored_hash)");
    println!("      Ok(true)  => past sequence intact; read only new files");
    println!("      Ok(false) => past sequence changed; rebuild all state from 0");
    println!("      Err(_)    => index unreadable; rebuild defensively");

    // --- Optional cleanup prompt -----------------------------------------
    prompt_cleanup(&temp_root, &watched_dir, &this_run_basenames);
}
