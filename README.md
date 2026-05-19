#### chrono_iter_dir_nav_rust

# Chrono Iter: Chronologial-Order Directory-File Navigation

For Posix (linux, MacOS, BSD, Unix, etc.):
Iterate chronologically through files in a local directory \
using a local-file-system on-file based lookup-table \
for chronological order.

However surprisingly, chronological file navigation cannot be done \
with normal Rust standard library or posix OS commands. Directory \
contents requests are unpredictable file-tree distributions. \
And mtime (time a file was modified) is not a default sort option. \
Also, storing many full-paths (or unpredictably many file paths!) \
in RAM (brute force) is infeasible and unsafe.

This vanilla rust (no 3rd party crates & no unsafe code blocks) \
allows chronological index file lookup, returning local \
absolute file paths by chronological order index search.

For longer names code can be revised.

Functions for debugging inspection and cleanup exist:
- purge_index_state(temp_root_dir: &Path)
- pub cold_build_summary
- pub append_summary

The goal has been memory-efficiency.

```rust
    // === Step 1: build / refresh the chronological lookup ===
    let update_summary = match update_index(&temp_root, &watched_dir) {
        Ok(summary) => summary,
        Err(error_code) => {
            eprintln!("update_index failed: {}", error_code.code());
            return;
        }
    };

    // === Step 2: Count files ===
    let total = match count_committed_files(&temp_root) {
        Ok(n) => n,
        Err(error_code) => {
            eprintln!("count_committed_files failed: {}", error_code.code());
            return;
        }
    };

    // === Step 3: Chrono Navigate to Index N ===
    let mut path_buffer = [0u8; MAX_FULL_PATH_LEN];
    let lookup_result = lookup_abs_file_path_at_mtime_chronological_index(
        &temp_root,
        requested_position,
        &mut path_buffer,
    );
```
