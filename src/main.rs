//! # Chrono-Iter Module — Mini Demo
//!
//! Demonstrates the two operations that constitute the chronological
//! lookup system:
//!
//!   1. Build (or refresh) the chronological lookup over a watched
//!      directory, using `update_index`.
//!   2. Return the absolute path of the file at a given chronological
//!      position, using `lookup_abs_file_path_at_mtime_chronological_index`.
//!
//! Layout on disk, relative to the cargo project root:
//!
//!   test/
//!     watched/   The directory being indexed (created if absent).
//!     temp/      The temp root used by the module (created if absent).
//!
//! Run with:   cargo run
//!
//! The default chronological position looked up is 0 (the
//! chronologically-earliest file). To look up a different position,
//! pass it as the first argument:
//!
//!   cargo run -- 3        # look up chronological position 3

use std::io::Write;
use std::path::{Path, PathBuf};

mod chrono_iter_module;

use chrono_iter_module::{
    MAX_FULL_PATH_LEN, UpdateOutcome, count_committed_files,
    lookup_abs_file_path_at_mtime_chronological_index, update_index,
};

/// Resolves `<cargo_manifest_dir>/test/<subdir>`, creating the
/// directory if absent. Demo-only convenience.
fn resolve_project_subdir(relative_subdir: &str) -> Result<PathBuf, std::io::Error> {
    let crate_root = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let mut full = PathBuf::from(crate_root);
    full.push("test");
    full.push(relative_subdir);
    std::fs::create_dir_all(&full)?;
    Ok(full)
}

/// Writes one uniquely-named file into `watched_dir` so each demo run
/// adds a new file (demonstrating that the index grows over time).
/// Returns the path of the file created.
fn add_one_growth_file(watched_dir: &Path) -> Result<PathBuf, std::io::Error> {
    let now_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let basename = format!("demo_{:020}.txt", now_nanos);
    let mut full = PathBuf::from(watched_dir);
    full.push(&basename);
    let mut handle = std::fs::File::create(&full)?;
    handle.write_all(b"demo content\n")?;
    handle.sync_all()?;
    Ok(full)
}

/// Parses an optional first CLI argument as a u64 chronological
/// position. Defaults to 0 if absent or malformed.
fn parse_requested_position_from_args() -> u64 {
    let mut args_iter = std::env::args();
    let _program_name = args_iter.next();
    match args_iter.next() {
        Some(arg) => arg.parse::<u64>().unwrap_or(0),
        None => 0,
    }
}

fn outcome_label(outcome: UpdateOutcome) -> &'static str {
    match outcome {
        UpdateOutcome::ColdBuildCompleted => "ColdBuildCompleted",
        UpdateOutcome::RebuiltDueToInconsistency => "RebuiltDueToInconsistency",
        UpdateOutcome::NoChangesDetected => "NoChangesDetected",
        UpdateOutcome::IncrementalAppendCompleted => "IncrementalAppendCompleted",
    }
}

fn main() {
    println!("chrono lookup demo");

    // --- Resolve project-local directories ---
    let watched_dir = match resolve_project_subdir("watched") {
        Ok(p) => p,
        Err(io_error) => {
            eprintln!("setup: cannot create test/watched: {}", io_error);
            return;
        }
    };
    let temp_root = match resolve_project_subdir("temp") {
        Ok(p) => p,
        Err(io_error) => {
            eprintln!("setup: cannot create test/temp: {}", io_error);
            return;
        }
    };
    println!("watched dir : {}", watched_dir.display());
    println!("temp root   : {}", temp_root.display());

    // --- Add one file so the demo has something to look up ---
    match add_one_growth_file(&watched_dir) {
        Ok(created) => println!("added file  : {}", created.display()),
        Err(io_error) => {
            println!("note: could not add a new file ({}); continuing", io_error);
        }
    }

    // === Step 1: build / refresh the chronological lookup ===
    let update_summary = match update_index(&temp_root, &watched_dir) {
        Ok(summary) => summary,
        Err(error_code) => {
            eprintln!("update_index failed: {}", error_code.code());
            return;
        }
    };
    println!(
        "update_index: outcome={} final_file_count={}",
        outcome_label(update_summary.outcome),
        update_summary.final_file_count,
    );

    // === Step 2: return the chrono-index file path at the requested position ===
    let total = match count_committed_files(&temp_root) {
        Ok(n) => n,
        Err(error_code) => {
            eprintln!("count_committed_files failed: {}", error_code.code());
            return;
        }
    };
    if total == 0 {
        println!("index is empty; nothing to look up");
        return;
    }

    let requested_position = parse_requested_position_from_args();
    if requested_position >= total {
        println!(
            "requested position {} is past end (file_count = {})",
            requested_position, total
        );
        return;
    }

    // === Step 3: Chrono Navigate to Index N ===
    let mut path_buffer = [0u8; MAX_FULL_PATH_LEN];
    let lookup_result = lookup_abs_file_path_at_mtime_chronological_index(
        &temp_root,
        requested_position,
        &mut path_buffer,
    );

    match lookup_result {
        Ok(Some(found)) => {
            let path_bytes = &path_buffer[..found.path_byte_length];
            println!(
                "chronological position {}:\n    path  = {}\n    mtime = {}.{:09}",
                requested_position,
                String::from_utf8_lossy(path_bytes),
                found.looked_up_file_mtime_sec,
                found.looked_up_file_mtime_nsec,
            );
        }
        Ok(None) => {
            println!("no file at position {}", requested_position);
        }
        Err(error_code) => {
            eprintln!("lookup failed: {}", error_code.code());
        }
    }
}
