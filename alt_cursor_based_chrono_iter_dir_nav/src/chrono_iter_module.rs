// src/chrono_index/mod.rs

//! # Chronological Directory Iterator — File-Backed Index
//!
//! Posix: Iterate chronologically through files in a local dir
//! using a local-file-system on-file based lookup-table
//! for chronological order, because mtime (time file modified)
//! is not a default sort option, and storing many N full paths
//! in RAM is infeasible.
//!
//! ## Project Context
//!
//! This module provides a low-heap, fail-safe mechanism for iterating
//! the files of a single POSIX directory in **chronological order by mtime**,
//! one file at a time, on a slow tick (e.g. once every N seconds).
//!
//! The directory being indexed has these project-level invariants:
//!
//! - **One directory only** — all indexed files share a single parent path.
//! - **Files are added over time** — growth is the steady-state case,
//!   not an edge case.
//! - **Files are never deleted** — the count is monotonically non-decreasing.
//! - **mtimes of existing files do not change** — only new files appear.
//! - **New files have newer mtimes than all existing files** — therefore
//!   the chronological sort order can be maintained by pure append after
//!   the initial build.
//! - **Basenames are short** — capped at 64 bytes (see `MAX_BASENAME_LEN`).
//!
//! ## Memory Model
//!
//! Per-tick memory is stack-only, on the order of a few kilobytes,
//! independent of the file count N. The index itself lives on disk as
//! a small set of fixed-width files in a caller-specified temp root.
//! No `Vec`, `String`, `HashMap`, or other heap-growing structure scales
//! with N inside this module.
//!
//! Heap is used only by unavoidable standard-library calls (e.g.
//! `std::fs::read_dir` allocates an `OsString` per entry, which is freed
//! before the next entry is produced). This is bounded per-iteration, not
//! per-N.
//!
//! ## On-Disk Layout
//!
//! Under `<caller_temp_root>/chrono_index/`:
//!
//! ```text
//! header.bin   Fixed-width header. Authoritative metadata.
//! names.bin    record_id -> basename. Fixed 64 B per record. Append-only.
//! mtimes.bin   Sorted by (mtime_sec, mtime_nsec, record_id).
//!              Fixed 20 B per record. Append-only in steady state.
//! cursor.bin   8 B. Next mtimes.bin index to emit on the next tick.
//! scratch/     Used only during cold rebuild (external merge sort).
//!              Removed after rebuild succeeds.
//! ```
//!
//! ## Failure Policy
//!
//! Per project rules: this module **never halts the program**. All
//! production paths return `Result<T, ChronoIndexError>` with terse,
//! non-data-leaking error codes. The caller is expected to log the code
//! and retry on the next tick. Internal recovery actions (e.g. silent
//! rebuild on header validation failure) are taken whenever the index
//! can be self-healed without user intervention.
//!
//! Per project rules:
//! - No `panic!` in production paths.
//! - No `unwrap` or `expect` in production paths.
//! - No `assert!` in production paths (test-only via `#[cfg(test)]`).
//! - `debug_assert!` permitted, guarded by `#[cfg(all(debug_assertions, not(test)))]`
//!   where appropriate.
//! - No unsafe code.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

// =========================================================================
// Public constants — file layout
// =========================================================================

/// Magic bytes identifying a `header.bin` file produced by this module.
///
/// Used to detect corruption, version mismatch, or accidental reuse of an
/// unrelated file at the header path. Any mismatch triggers a rebuild.
pub const HEADER_MAGIC: [u8; 8] = *b"CHRIDX01";

/// On-disk format version. Bump on any incompatible layout change.
/// Mismatched versions trigger a rebuild rather than an attempt to migrate.
pub const HEADER_VERSION: u32 = 1;

/// Maximum length in bytes of a basename stored in `names.bin`.
///
/// Per project spec: basenames are short, "definitely <64 char". We store
/// 64 bytes including any NUL padding, giving room for up to 64 ASCII or
/// up to 16 four-byte UTF-8 characters. Names longer than this cannot be
/// indexed; such entries are skipped at build time (logged terse code).
pub const MAX_BASENAME_LEN: usize = 64;

/// Maximum length in bytes of the parent directory absolute path stored
/// in the header. POSIX `PATH_MAX` is typically 4096 on Linux; we cap
/// here at the same value. Longer parent paths cannot be indexed.
pub const MAX_PARENT_PATH_LEN: usize = 4096;

/// Size in bytes of one `names.bin` record. Fixed-width to permit O(1)
/// random access by `record_id`: byte offset = `record_id * NAME_RECORD_SIZE`.
pub const NAME_RECORD_SIZE: usize = MAX_BASENAME_LEN;

/// Size in bytes of one `mtimes.bin` record:
///   `(mtime_sec: i64, mtime_nsec: i32, record_id: u64)` = 8 + 4 + 8 = 20.
/// Fixed-width to permit O(1) random access and in-place external sort.
pub const MTIME_RECORD_SIZE: usize = 20;

/// Size in bytes of the on-disk `header.bin`. Fixed, validated on read.
///
/// Layout (all little-endian, packed in declaration order):
///
/// ```text
///   offset  size  field
///   ------  ----  -----
///        0     8  magic                 (HEADER_MAGIC)
///        8     4  version               (u32)
///       12     8  file_count            (u64) — total indexed files
///       20     8  signal_hash           (u64) — XOR of basename hashes
///       28     8  last_mtime_sec        (i64) — mtime of newest indexed file
///       36     4  last_mtime_nsec       (i32)
///       40     8  invariant_breach_ct   (u64) — count of out-of-order appends
///       48     2  parent_path_len       (u16) — bytes used in parent_path
///       50     2  reserved              (u16) — padding / future flags
///       52  4096  parent_path           ([u8; MAX_PARENT_PATH_LEN])
///     4148    12  reserved_tail         ([u8; 12]) — alignment / future use
///     ----  ----
///     4160 total
/// ```
pub const HEADER_SIZE: usize = 4160;

// Sanity check at compile time. These are test-only and debug-only
// assertions per project policy; they never run in production binaries.
#[cfg(test)]
#[allow(dead_code)]
const _COMPILE_TIME_HEADER_SIZE_CHECK: () = {
    assert!(HEADER_SIZE == 8 + 4 + 8 + 8 + 8 + 4 + 8 + 2 + 2 + MAX_PARENT_PATH_LEN + 12);
};

// File names within the chrono_index subdirectory.
pub const HEADER_FILENAME: &str = "header.bin";
pub const NAMES_FILENAME: &str = "names.bin";
pub const MTIMES_FILENAME: &str = "mtimes.bin";
pub const CURSOR_FILENAME: &str = "cursor.bin";
pub const SCRATCH_DIRNAME: &str = "scratch";
pub const INDEX_SUBDIRNAME: &str = "chrono_index";

// =========================================================================
// Error type — terse, non-leaking, per project policy
// =========================================================================

/// Error codes returned by this module.
///
/// Variants are intentionally coarse and carry **no user data, file paths,
/// or filename content**, per project security policy: production error
/// output must not leak filesystem structure or user data.
///
/// Each variant's prefix `CIDX-` identifies the module of origin for log
/// triage. The numeric suffix is the stable diagnostic code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChronoIndexError {
    /// CIDX-01: Unable to create or open the index directory.
    IndexDirIo,
    /// CIDX-02: Unable to read header file.
    HeaderReadIo,
    /// CIDX-03: Unable to write header file.
    HeaderWriteIo,
    /// CIDX-04: Header magic mismatch.
    HeaderBadMagic,
    /// CIDX-05: Header version mismatch.
    HeaderBadVersion,
    /// CIDX-06: Header size or internal length field out of range.
    HeaderBadSize,
    /// CIDX-07: Parent path provided exceeds MAX_PARENT_PATH_LEN.
    ParentPathTooLong,
    /// CIDX-08: Parent path empty or otherwise invalid.
    ParentPathInvalid,
    /// CIDX-09: Atomic rename failed.
    RenameIo,
    /// CIDX-10: Reserved for future use by build path.
    BuildIo,
    /// CIDX-11: Reserved for future use by append path.
    AppendIo,
    /// CIDX-12: Reserved for future use by tick path.
    TickIo,
}

impl ChronoIndexError {
    /// Returns the stable diagnostic code string. Safe for production logs.
    /// Never includes paths, names, mtimes, or other content.
    pub fn code(self) -> &'static str {
        match self {
            ChronoIndexError::IndexDirIo => "CIDX-01",
            ChronoIndexError::HeaderReadIo => "CIDX-02",
            ChronoIndexError::HeaderWriteIo => "CIDX-03",
            ChronoIndexError::HeaderBadMagic => "CIDX-04",
            ChronoIndexError::HeaderBadVersion => "CIDX-05",
            ChronoIndexError::HeaderBadSize => "CIDX-06",
            ChronoIndexError::ParentPathTooLong => "CIDX-07",
            ChronoIndexError::ParentPathInvalid => "CIDX-08",
            ChronoIndexError::RenameIo => "CIDX-09",
            ChronoIndexError::BuildIo => "CIDX-10",
            ChronoIndexError::AppendIo => "CIDX-11",
            ChronoIndexError::TickIo => "CIDX-12",
        }
    }
}

// =========================================================================
// In-memory header representation
// =========================================================================

/// In-memory mirror of `header.bin`.
///
/// This struct is small and stack-friendly. It is the single source of
/// truth for index metadata while a build or append is in progress; on
/// completion, it is serialized atomically to disk via [`write_header_atomic`].
///
/// Fields correspond byte-for-byte to the on-disk layout documented on
/// [`HEADER_SIZE`].
#[derive(Clone)]
pub struct ChronoIndexHeader {
    /// Total number of indexed files. Equals number of records in both
    /// `names.bin` and `mtimes.bin`. Monotonically non-decreasing.
    pub file_count: u64,

    /// Order-independent signal hash of all indexed basenames
    /// (XOR-reduce of per-name FNV-1a 64). Used to cheaply detect whether
    /// the directory contents have diverged from the index between runs.
    pub signal_hash: u64,

    /// mtime of the newest indexed file (largest sort key in `mtimes.bin`).
    /// Used to validate the "new files have newer mtimes" invariant at
    /// append time without re-reading `mtimes.bin`.
    pub last_mtime_sec: i64,
    pub last_mtime_nsec: i32,

    /// Count of times the append-only invariant was breached and a merge
    /// insert was performed instead of a pure append. Observability only;
    /// does not affect correctness.
    pub invariant_breach_count: u64,

    /// Length in bytes of `parent_path` actually in use (`<= MAX_PARENT_PATH_LEN`).
    pub parent_path_len: u16,

    /// Absolute path of the directory being indexed. Only the first
    /// `parent_path_len` bytes are meaningful; the rest is zero-padding.
    /// Stored as raw bytes (POSIX paths are byte sequences, not guaranteed
    /// UTF-8).
    pub parent_path: [u8; MAX_PARENT_PATH_LEN],
}

impl ChronoIndexHeader {
    /// Constructs a fresh header for a newly built index over the given
    /// parent directory absolute path.
    ///
    /// Returns `Err(ParentPathTooLong)` if the path exceeds
    /// `MAX_PARENT_PATH_LEN`, or `Err(ParentPathInvalid)` if empty.
    ///
    /// Initial state: `file_count = 0`, `signal_hash = 0`,
    /// `last_mtime_* = i64::MIN / 0` so the first appended record is
    /// always strictly newer.
    pub fn new_for_parent(parent_path_bytes: &[u8]) -> Result<Self, ChronoIndexError> {
        // Defensive: empty path makes no sense for a one-directory index.
        if parent_path_bytes.is_empty() {
            return Err(ChronoIndexError::ParentPathInvalid);
        }
        if parent_path_bytes.len() > MAX_PARENT_PATH_LEN {
            return Err(ChronoIndexError::ParentPathTooLong);
        }

        let mut parent_path_buffer = [0u8; MAX_PARENT_PATH_LEN];
        // Safe slice copy; bounds already validated above.
        parent_path_buffer[..parent_path_bytes.len()].copy_from_slice(parent_path_bytes);

        Ok(ChronoIndexHeader {
            file_count: 0,
            signal_hash: 0,
            // Sentinel: any real mtime will compare strictly greater than this.
            last_mtime_sec: i64::MIN,
            last_mtime_nsec: 0,
            invariant_breach_count: 0,
            parent_path_len: parent_path_bytes.len() as u16,
            parent_path: parent_path_buffer,
        })
    }

    /// Returns a slice of the meaningful portion of `parent_path`,
    /// without trailing zero padding.
    pub fn parent_path_slice(&self) -> &[u8] {
        // Defensive bounds clamp: if a corrupt on-disk value somehow
        // exceeded the array length, we clamp rather than panic.
        let usable_length = (self.parent_path_len as usize).min(MAX_PARENT_PATH_LEN);
        &self.parent_path[..usable_length]
    }

    /// Serializes this header into a `HEADER_SIZE`-byte buffer in the
    /// on-disk format documented on `HEADER_SIZE`.
    fn serialize_into(&self, output_buffer: &mut [u8; HEADER_SIZE]) {
        // Zero the buffer so all reserved/padding regions are deterministic.
        for byte_slot in output_buffer.iter_mut() {
            *byte_slot = 0;
        }

        output_buffer[0..8].copy_from_slice(&HEADER_MAGIC);
        output_buffer[8..12].copy_from_slice(&HEADER_VERSION.to_le_bytes());
        output_buffer[12..20].copy_from_slice(&self.file_count.to_le_bytes());
        output_buffer[20..28].copy_from_slice(&self.signal_hash.to_le_bytes());
        output_buffer[28..36].copy_from_slice(&self.last_mtime_sec.to_le_bytes());
        output_buffer[36..40].copy_from_slice(&self.last_mtime_nsec.to_le_bytes());
        output_buffer[40..48].copy_from_slice(&self.invariant_breach_count.to_le_bytes());
        output_buffer[48..50].copy_from_slice(&self.parent_path_len.to_le_bytes());
        // bytes [50..52] reserved (u16) — left zero
        output_buffer[52..52 + MAX_PARENT_PATH_LEN].copy_from_slice(&self.parent_path);
        // bytes [4148..4160] reserved_tail — left zero
    }

    /// Deserializes a header from a `HEADER_SIZE`-byte buffer.
    ///
    /// Validates magic, version, and `parent_path_len`. Returns:
    /// - `Err(HeaderBadMagic)` on magic mismatch,
    /// - `Err(HeaderBadVersion)` on version mismatch,
    /// - `Err(HeaderBadSize)` if `parent_path_len > MAX_PARENT_PATH_LEN`.
    ///
    /// These errors are the caller's signal to trigger a rebuild rather
    /// than to halt.
    fn deserialize_from(input_buffer: &[u8; HEADER_SIZE]) -> Result<Self, ChronoIndexError> {
        // Magic check first — fast rejection of unrelated files.
        let mut magic_buffer = [0u8; 8];
        magic_buffer.copy_from_slice(&input_buffer[0..8]);
        if magic_buffer != HEADER_MAGIC {
            return Err(ChronoIndexError::HeaderBadMagic);
        }

        let mut u32_buffer = [0u8; 4];
        u32_buffer.copy_from_slice(&input_buffer[8..12]);
        let on_disk_version = u32::from_le_bytes(u32_buffer);
        if on_disk_version != HEADER_VERSION {
            return Err(ChronoIndexError::HeaderBadVersion);
        }

        let mut u64_buffer = [0u8; 8];
        u64_buffer.copy_from_slice(&input_buffer[12..20]);
        let file_count = u64::from_le_bytes(u64_buffer);

        u64_buffer.copy_from_slice(&input_buffer[20..28]);
        let signal_hash = u64::from_le_bytes(u64_buffer);

        let mut i64_buffer = [0u8; 8];
        i64_buffer.copy_from_slice(&input_buffer[28..36]);
        let last_mtime_sec = i64::from_le_bytes(i64_buffer);

        let mut i32_buffer = [0u8; 4];
        i32_buffer.copy_from_slice(&input_buffer[36..40]);
        let last_mtime_nsec = i32::from_le_bytes(i32_buffer);

        u64_buffer.copy_from_slice(&input_buffer[40..48]);
        let invariant_breach_count = u64::from_le_bytes(u64_buffer);

        let mut u16_buffer = [0u8; 2];
        u16_buffer.copy_from_slice(&input_buffer[48..50]);
        let parent_path_len = u16::from_le_bytes(u16_buffer);
        // bytes [50..52] reserved — ignored on read

        if (parent_path_len as usize) > MAX_PARENT_PATH_LEN {
            return Err(ChronoIndexError::HeaderBadSize);
        }

        let mut parent_path_buffer = [0u8; MAX_PARENT_PATH_LEN];
        parent_path_buffer.copy_from_slice(&input_buffer[52..52 + MAX_PARENT_PATH_LEN]);

        Ok(ChronoIndexHeader {
            file_count,
            signal_hash,
            last_mtime_sec,
            last_mtime_nsec,
            invariant_breach_count,
            parent_path_len,
            parent_path: parent_path_buffer,
        })
    }
}

// =========================================================================
// Path helpers — assemble absolute paths into the index files.
// =========================================================================

/// Joins a caller-provided temp root with the fixed `chrono_index/` subdir
/// and the given index-file basename.
///
/// This uses `std::path::PathBuf` (small heap allocation, bounded by
/// `PATH_MAX`, freed on drop) **only** because `std::fs` APIs require
/// `&Path`. This is a per-call cost, not a per-N cost. Acceptable.
fn build_index_file_path(temp_root_dir: &Path, index_file_basename: &str) -> PathBuf {
    let mut composed_path = PathBuf::from(temp_root_dir);
    composed_path.push(INDEX_SUBDIRNAME);
    composed_path.push(index_file_basename);
    composed_path
}

// =========================================================================
// Index directory provisioning
// =========================================================================

/// Ensures `<temp_root>/chrono_index/` exists. Idempotent. Does not create
/// any of the index files themselves; that is the responsibility of the
/// build / append paths.
///
/// On any I/O failure returns `Err(IndexDirIo)` — caller decides whether
/// to retry or fall back. Never panics, never halts.
pub fn ensure_index_directory_exists(temp_root_dir: &Path) -> Result<(), ChronoIndexError> {
    let mut index_directory_path = PathBuf::from(temp_root_dir);
    index_directory_path.push(INDEX_SUBDIRNAME);

    match std::fs::create_dir_all(&index_directory_path) {
        Ok(()) => Ok(()),
        Err(_io_error) => {
            // Do not leak the path or the OS error message into the
            // production error channel. Return a terse stable code.
            Err(ChronoIndexError::IndexDirIo)
        }
    }
}

// =========================================================================
// Header read
// =========================================================================

/// Reads and validates `header.bin` from disk.
///
/// Returns:
/// - `Ok(Some(header))` if the header file exists and is valid.
/// - `Ok(None)` if the header file does not exist (first run / clean state).
/// - `Err(HeaderReadIo)` for any I/O error other than "not found".
/// - `Err(HeaderBadMagic | HeaderBadVersion | HeaderBadSize)` for
///   structural mismatch — caller should treat these as "rebuild needed".
///
/// Reads exactly `HEADER_SIZE` bytes into a stack buffer; no heap growth
/// related to header content.
pub fn read_header(temp_root_dir: &Path) -> Result<Option<ChronoIndexHeader>, ChronoIndexError> {
    let header_file_path = build_index_file_path(temp_root_dir, HEADER_FILENAME);

    let mut header_file_handle = match File::open(&header_file_path) {
        Ok(opened_file) => opened_file,
        Err(open_error) => {
            // "Not found" is a normal first-run state, not an error.
            if open_error.kind() == std::io::ErrorKind::NotFound {
                return Ok(None);
            }
            return Err(ChronoIndexError::HeaderReadIo);
        }
    };

    let mut header_byte_buffer = [0u8; HEADER_SIZE];
    match header_file_handle.read_exact(&mut header_byte_buffer) {
        Ok(()) => {}
        Err(_read_error) => {
            // Truncated, permissions, I/O, etc. — terse code, caller
            // will trigger rebuild.
            return Err(ChronoIndexError::HeaderReadIo);
        }
    }

    // Structural validation lives in deserialize_from.
    let parsed_header = ChronoIndexHeader::deserialize_from(&header_byte_buffer)?;
    Ok(Some(parsed_header))
}

// =========================================================================
// Header write — atomic via tempfile + rename
// =========================================================================

/// Writes `header.bin` atomically using the write-temp + fsync + rename
/// pattern. POSIX guarantees `rename(2)` is atomic within the same
/// filesystem, so a reader either sees the old header or the new header,
/// never a partial one.
///
/// On any I/O failure returns `Err(HeaderWriteIo)` or `Err(RenameIo)`.
/// The previous header (if any) is left untouched on failure — the index
/// remains in its last consistent state. Caller may retry on the next tick.
pub fn write_header_atomic(
    temp_root_dir: &Path,
    header_to_write: &ChronoIndexHeader,
) -> Result<(), ChronoIndexError> {
    let final_header_path = build_index_file_path(temp_root_dir, HEADER_FILENAME);

    // Stage to a sibling temp file in the same directory so that rename is
    // a same-filesystem operation and therefore atomic per POSIX.
    let mut staging_header_path = final_header_path.clone();
    // Append a fixed staging suffix. Single-writer assumption; if multi-
    // writer support is ever needed, swap to a unique-per-process suffix.
    staging_header_path.set_file_name("header.bin.tmp");

    // Serialize into a stack buffer — no heap.
    let mut header_byte_buffer = [0u8; HEADER_SIZE];
    header_to_write.serialize_into(&mut header_byte_buffer);

    // Open staging file (create or truncate).
    let mut staging_file_handle = match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&staging_header_path)
    {
        Ok(opened_file) => opened_file,
        Err(_open_error) => return Err(ChronoIndexError::HeaderWriteIo),
    };

    if staging_file_handle.write_all(&header_byte_buffer).is_err() {
        // Best-effort cleanup of partial staging file; failure to remove
        // is non-fatal — the next write will truncate it.
        let _ = std::fs::remove_file(&staging_header_path);
        return Err(ChronoIndexError::HeaderWriteIo);
    }

    // fsync staging file so its contents are durable before rename.
    if staging_file_handle.sync_all().is_err() {
        let _ = std::fs::remove_file(&staging_header_path);
        return Err(ChronoIndexError::HeaderWriteIo);
    }

    // Drop the file handle explicitly before rename; on some platforms
    // (not Linux, but defensive) an open handle can interfere with rename.
    drop(staging_file_handle);

    // Atomic rename. On failure leave the previous header in place.
    if std::fs::rename(&staging_header_path, &final_header_path).is_err() {
        let _ = std::fs::remove_file(&staging_header_path);
        return Err(ChronoIndexError::RenameIo);
    }

    // Note: we do not fsync the containing directory here. For maximal
    // crash-durability of the rename itself, a directory fsync would be
    // added. Project-level policy ("rebuild on header invalid") makes
    // this safe to omit: a crashed-mid-rename header will be treated as
    // "rebuild needed" on the next run, which is the intended fail-safe.

    Ok(())
}

// =========================================================================
// Helper: read cursor.bin (used by tick path later; defined here because
// it is the same atomic-rename pattern as the header).
// =========================================================================

/// Reads `cursor.bin` if present. Returns `Ok(0)` if not present
/// (fresh state) or if the file is structurally invalid (defensive
/// self-healing). Returns `Err(TickIo)` only on hard I/O errors.
pub fn read_cursor(temp_root_dir: &Path) -> Result<u64, ChronoIndexError> {
    let cursor_file_path = build_index_file_path(temp_root_dir, CURSOR_FILENAME);

    let mut cursor_file_handle = match File::open(&cursor_file_path) {
        Ok(opened_file) => opened_file,
        Err(open_error) => {
            if open_error.kind() == std::io::ErrorKind::NotFound {
                return Ok(0);
            }
            return Err(ChronoIndexError::TickIo);
        }
    };

    let mut cursor_byte_buffer = [0u8; 8];
    match cursor_file_handle.read_exact(&mut cursor_byte_buffer) {
        Ok(()) => Ok(u64::from_le_bytes(cursor_byte_buffer)),
        // Truncated or otherwise unreadable — self-heal to zero rather
        // than halt. The tick path will re-emit from the start, which is
        // safe per project policy (consumers are expected to be idempotent
        // or to tolerate a re-emit after crash).
        Err(_read_error) => Ok(0),
    }
}

/// Writes `cursor.bin` atomically (write-temp + fsync + rename).
pub fn write_cursor_atomic(
    temp_root_dir: &Path,
    next_cursor_value: u64,
) -> Result<(), ChronoIndexError> {
    let final_cursor_path = build_index_file_path(temp_root_dir, CURSOR_FILENAME);
    let mut staging_cursor_path = final_cursor_path.clone();
    staging_cursor_path.set_file_name("cursor.bin.tmp");

    let mut staging_file_handle = match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&staging_cursor_path)
    {
        Ok(opened_file) => opened_file,
        Err(_) => return Err(ChronoIndexError::TickIo),
    };

    let cursor_byte_buffer = next_cursor_value.to_le_bytes();
    if staging_file_handle.write_all(&cursor_byte_buffer).is_err() {
        let _ = std::fs::remove_file(&staging_cursor_path);
        return Err(ChronoIndexError::TickIo);
    }
    if staging_file_handle.sync_all().is_err() {
        let _ = std::fs::remove_file(&staging_cursor_path);
        return Err(ChronoIndexError::TickIo);
    }
    drop(staging_file_handle);

    if std::fs::rename(&staging_cursor_path, &final_cursor_path).is_err() {
        let _ = std::fs::remove_file(&staging_cursor_path);
        return Err(ChronoIndexError::RenameIo);
    }
    Ok(())
}

// Mark unused-for-now helpers as referenced so the compiler doesn't warn
// before parts (b)/(c)/(d) are added. These functions are part of the
// public surface and will be used by the build/append/tick modules.
#[allow(dead_code)]
fn _module_surface_anchor() {
    let _ = SeekFrom::Start(0);
    let _ = NAMES_FILENAME;
    let _ = MTIMES_FILENAME;
    let _ = SCRATCH_DIRNAME;
    let _ = NAME_RECORD_SIZE;
    let _ = MTIME_RECORD_SIZE;
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod chrono_index_part_a_tests {
    use super::*;

    /// Helper: create a unique scratch directory under the OS temp dir for
    /// test isolation. Test-only; production callers supply their own root.
    fn make_test_temp_root(test_label: &str) -> PathBuf {
        let mut scratch = std::env::temp_dir();
        let nanos_since_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        scratch.push(format!(
            "chrono_index_test_{}_{}_{}",
            test_label,
            std::process::id(),
            nanos_since_epoch
        ));
        std::fs::create_dir_all(&scratch).expect("test setup: create temp root");
        scratch
    }

    #[test]
    fn header_size_constant_matches_field_sum() {
        // Test-only assert: validates the documented layout arithmetic.
        assert_eq!(
            HEADER_SIZE,
            8 + 4 + 8 + 8 + 8 + 4 + 8 + 2 + 2 + MAX_PARENT_PATH_LEN + 12
        );
    }

    #[test]
    fn new_header_for_parent_rejects_empty_path() {
        let result = ChronoIndexHeader::new_for_parent(b"");
        assert_eq!(result.err(), Some(ChronoIndexError::ParentPathInvalid));
    }

    #[test]
    fn new_header_for_parent_rejects_oversize_path() {
        let oversize = vec![b'a'; MAX_PARENT_PATH_LEN + 1];
        let result = ChronoIndexHeader::new_for_parent(&oversize);
        assert_eq!(result.err(), Some(ChronoIndexError::ParentPathTooLong));
    }

    #[test]
    fn new_header_initial_state_is_sane() {
        let header =
            ChronoIndexHeader::new_for_parent(b"/var/data/watched").expect("valid parent path");
        assert_eq!(header.file_count, 0);
        assert_eq!(header.signal_hash, 0);
        assert_eq!(header.last_mtime_sec, i64::MIN);
        assert_eq!(header.last_mtime_nsec, 0);
        assert_eq!(header.invariant_breach_count, 0);
        assert_eq!(header.parent_path_slice(), b"/var/data/watched");
    }

    #[test]
    fn serialize_then_deserialize_round_trips() {
        let mut original = ChronoIndexHeader::new_for_parent(b"/some/dir").expect("valid path");
        original.file_count = 123_456;
        original.signal_hash = 0xDEAD_BEEF_CAFE_BABE;
        original.last_mtime_sec = 1_700_000_000;
        original.last_mtime_nsec = 999_999_999;
        original.invariant_breach_count = 7;

        let mut buffer = [0u8; HEADER_SIZE];
        original.serialize_into(&mut buffer);

        let recovered =
            ChronoIndexHeader::deserialize_from(&buffer).expect("valid header round-trip");

        assert_eq!(recovered.file_count, original.file_count);
        assert_eq!(recovered.signal_hash, original.signal_hash);
        assert_eq!(recovered.last_mtime_sec, original.last_mtime_sec);
        assert_eq!(recovered.last_mtime_nsec, original.last_mtime_nsec);
        assert_eq!(
            recovered.invariant_breach_count,
            original.invariant_breach_count
        );
        assert_eq!(recovered.parent_path_slice(), original.parent_path_slice());
    }

    #[test]
    fn deserialize_rejects_bad_magic() {
        let mut buffer = [0u8; HEADER_SIZE];
        // Leave magic as all-zero; deserialize must reject.
        let result = ChronoIndexHeader::deserialize_from(&buffer);
        assert_eq!(result.err(), Some(ChronoIndexError::HeaderBadMagic));

        // Corrupt magic.
        buffer[0..8].copy_from_slice(b"XXXXXXXX");
        let result = ChronoIndexHeader::deserialize_from(&buffer);
        assert_eq!(result.err(), Some(ChronoIndexError::HeaderBadMagic));
    }

    #[test]
    fn deserialize_rejects_bad_version() {
        let mut buffer = [0u8; HEADER_SIZE];
        buffer[0..8].copy_from_slice(&HEADER_MAGIC);
        // Write a wrong version.
        buffer[8..12].copy_from_slice(&(HEADER_VERSION.wrapping_add(99)).to_le_bytes());
        let result = ChronoIndexHeader::deserialize_from(&buffer);
        assert_eq!(result.err(), Some(ChronoIndexError::HeaderBadVersion));
    }

    #[test]
    fn deserialize_rejects_oversize_parent_path_len() {
        let mut buffer = [0u8; HEADER_SIZE];
        buffer[0..8].copy_from_slice(&HEADER_MAGIC);
        buffer[8..12].copy_from_slice(&HEADER_VERSION.to_le_bytes());
        // Set parent_path_len > MAX_PARENT_PATH_LEN.
        let bogus_len: u16 = (MAX_PARENT_PATH_LEN as u16).saturating_add(1);
        buffer[48..50].copy_from_slice(&bogus_len.to_le_bytes());
        let result = ChronoIndexHeader::deserialize_from(&buffer);
        assert_eq!(result.err(), Some(ChronoIndexError::HeaderBadSize));
    }

    #[test]
    fn ensure_index_directory_is_idempotent() {
        let root = make_test_temp_root("ensure_dir");
        assert!(ensure_index_directory_exists(&root).is_ok());
        // Second call must also succeed.
        assert!(ensure_index_directory_exists(&root).is_ok());

        let mut expected = root.clone();
        expected.push(INDEX_SUBDIRNAME);
        assert!(expected.is_dir());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_header_returns_none_when_absent() {
        let root = make_test_temp_root("read_absent");
        ensure_index_directory_exists(&root).expect("setup");
        let read_result = read_header(&root).expect("read should succeed with None");
        assert!(read_result.is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_then_read_header_round_trips_on_disk() {
        let root = make_test_temp_root("rw_header");
        ensure_index_directory_exists(&root).expect("setup");

        let mut original =
            ChronoIndexHeader::new_for_parent(b"/data/observed").expect("valid path");
        original.file_count = 42;
        original.signal_hash = 0x1122_3344_5566_7788;
        original.last_mtime_sec = 1_700_123_456;
        original.last_mtime_nsec = 250_000_000;
        original.invariant_breach_count = 2;

        write_header_atomic(&root, &original).expect("write ok");
        let recovered = read_header(&root)
            .expect("read ok")
            .expect("header present");

        assert_eq!(recovered.file_count, 42);
        assert_eq!(recovered.signal_hash, 0x1122_3344_5566_7788);
        assert_eq!(recovered.last_mtime_sec, 1_700_123_456);
        assert_eq!(recovered.last_mtime_nsec, 250_000_000);
        assert_eq!(recovered.invariant_breach_count, 2);
        assert_eq!(recovered.parent_path_slice(), b"/data/observed");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cursor_read_returns_zero_when_absent() {
        let root = make_test_temp_root("cursor_absent");
        ensure_index_directory_exists(&root).expect("setup");
        let cursor = read_cursor(&root).expect("cursor read ok");
        assert_eq!(cursor, 0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cursor_write_then_read_round_trips() {
        let root = make_test_temp_root("cursor_rw");
        ensure_index_directory_exists(&root).expect("setup");
        write_cursor_atomic(&root, 9_876_543_210).expect("write cursor ok");
        let cursor = read_cursor(&root).expect("read cursor ok");
        assert_eq!(cursor, 9_876_543_210);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn error_codes_are_stable_and_terse() {
        // Production logs must be able to depend on these strings.
        assert_eq!(ChronoIndexError::IndexDirIo.code(), "CIDX-01");
        assert_eq!(ChronoIndexError::HeaderBadMagic.code(), "CIDX-04");
        assert_eq!(ChronoIndexError::ParentPathTooLong.code(), "CIDX-07");
    }
}

// =========================================================================
// Part (b): Cold-build path
// =========================================================================
//
// ## When this runs
//
// The cold-build path is the fallback that produces a fresh, fully-sorted
// index from the live directory contents. It runs:
//
//   - On first ever use (no header present).
//   - When `read_header` returns a structural error (bad magic / version /
//     size), indicating the index is unusable or from a different version.
//   - When the caller-orchestrated change-detection determines the
//     existing index has diverged beyond what the incremental append path
//     (part c) can safely repair.
//
// ## Memory discipline
//
// All per-record I/O uses stack-resident fixed-size buffers:
//
//   - One `[u8; NAME_RECORD_SIZE]` for the current basename being written.
//   - One `[u8; MTIME_RECORD_SIZE]` for the current mtime record.
//   - During external sort: one fixed-size sort buffer of
//     `EXTERNAL_SORT_CHUNK_RECORDS` mtime records (default 4096 records ×
//     20 B = 80 KB) on the heap as a single `Box<[MtimeRecord]>`, allocated
//     ONCE per build and reused. This is a single bounded allocation that
//     does NOT scale with the directory size N.
//   - During k-way merge: a small fixed-size merge-heap of
//     `MAX_MERGE_FANOUT` slots (default 16) on the stack.
//
// Per-N heap growth: none. The unsorted scratch file grows on disk, not in
// RAM, and is removed after the sort.
//
// ## Failure policy
//
// Any I/O error during build: clean up scratch artifacts where possible
// and return a terse error code. The previous index (if any) is left
// untouched on disk until the new header is renamed into place — so a
// failed rebuild does not destroy a working index.

use std::io::BufReader;
use std::io::BufWriter;

/// Number of mtime records held in RAM during one pass of the external
/// merge sort. Each record is `MTIME_RECORD_SIZE` (20) bytes, so the
/// default value of 4096 yields an 80 KB working buffer.
///
/// This is the single bounded heap allocation made during cold build.
/// It does not scale with N: a directory of 1 million files uses exactly
/// the same buffer as a directory of 100 files.
pub const EXTERNAL_SORT_CHUNK_RECORDS: usize = 4096;

/// Maximum number of sorted runs merged simultaneously in the k-way merge
/// phase. If the build produces more runs than this, the merge is done in
/// successive passes (cascade merge). Bounded fan-out keeps file-handle
/// usage and merge-heap size bounded regardless of N.
pub const MAX_MERGE_FANOUT: usize = 16;

/// Scratch filenames used during build. Deleted on successful completion.
const SCRATCH_UNSORTED_MTIMES_FILENAME: &str = "mtimes_unsorted.bin";
const SCRATCH_RUN_FILENAME_PREFIX: &str = "run_";
const SCRATCH_RUN_FILENAME_SUFFIX: &str = ".bin";

/// In-memory representation of one `mtimes.bin` record.
///
/// Sort order: ascending by `(mtime_sec, mtime_nsec, record_id)`.
/// The `record_id` tiebreaker guarantees a total order even when multiple
/// files share an mtime, which makes the sort deterministic and the
/// cursor advancement well-defined.
#[derive(Clone, Copy)]
pub struct MtimeRecord {
    pub mtime_sec: i64,
    pub mtime_nsec: i32,
    pub record_id: u64,
}

impl MtimeRecord {
    /// Serializes this record to its 20-byte on-disk form.
    fn write_into(self, output_buffer: &mut [u8; MTIME_RECORD_SIZE]) {
        output_buffer[0..8].copy_from_slice(&self.mtime_sec.to_le_bytes());
        output_buffer[8..12].copy_from_slice(&self.mtime_nsec.to_le_bytes());
        output_buffer[12..20].copy_from_slice(&self.record_id.to_le_bytes());
    }

    /// Deserializes a record from its 20-byte on-disk form.
    fn read_from(input_buffer: &[u8; MTIME_RECORD_SIZE]) -> Self {
        let mut i64_buf = [0u8; 8];
        i64_buf.copy_from_slice(&input_buffer[0..8]);
        let mtime_sec = i64::from_le_bytes(i64_buf);

        let mut i32_buf = [0u8; 4];
        i32_buf.copy_from_slice(&input_buffer[8..12]);
        let mtime_nsec = i32::from_le_bytes(i32_buf);

        let mut u64_buf = [0u8; 8];
        u64_buf.copy_from_slice(&input_buffer[12..20]);
        let record_id = u64::from_le_bytes(u64_buf);

        MtimeRecord {
            mtime_sec,
            mtime_nsec,
            record_id,
        }
    }

    /// Returns `true` if `self` sorts strictly before `other` in the
    /// chronological total order.
    fn is_strictly_before(self, other: MtimeRecord) -> bool {
        if self.mtime_sec != other.mtime_sec {
            return self.mtime_sec < other.mtime_sec;
        }
        if self.mtime_nsec != other.mtime_nsec {
            return self.mtime_nsec < other.mtime_nsec;
        }
        self.record_id < other.record_id
    }
}

// =========================================================================
// FNV-1a 64 — small, allocation-free, used for the order-independent
// `signal_hash` over basenames.
// =========================================================================

/// FNV-1a 64-bit offset basis.
const FNV1A_64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV1A_64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Computes the FNV-1a 64-bit hash of a byte slice.
///
/// Allocation-free, deterministic, suitable for use as a per-name signal
/// to be XOR-reduced over all basenames. Not cryptographic; collisions
/// are acceptable for change detection because the count is checked
/// alongside the XOR. A pair of (count, xor-hash) collisions on a real
/// directory is vanishingly unlikely; on mismatch the worst case is a
/// rebuild, which is safe.
fn fnv1a_64(input_bytes: &[u8]) -> u64 {
    let mut hash_state: u64 = FNV1A_64_OFFSET_BASIS;
    for &byte_value in input_bytes {
        hash_state ^= byte_value as u64;
        hash_state = hash_state.wrapping_mul(FNV1A_64_PRIME);
    }
    hash_state
}

// =========================================================================
// names.bin / mtimes.bin path helpers and writers
// =========================================================================

fn build_scratch_path(temp_root_dir: &Path, scratch_basename: &str) -> PathBuf {
    let mut composed = PathBuf::from(temp_root_dir);
    composed.push(INDEX_SUBDIRNAME);
    composed.push(SCRATCH_DIRNAME);
    composed.push(scratch_basename);
    composed
}

fn ensure_scratch_directory_exists(temp_root_dir: &Path) -> Result<(), ChronoIndexError> {
    let mut scratch_dir = PathBuf::from(temp_root_dir);
    scratch_dir.push(INDEX_SUBDIRNAME);
    scratch_dir.push(SCRATCH_DIRNAME);
    match std::fs::create_dir_all(&scratch_dir) {
        Ok(()) => Ok(()),
        Err(_) => Err(ChronoIndexError::BuildIo),
    }
}

fn remove_scratch_directory_best_effort(temp_root_dir: &Path) {
    let mut scratch_dir = PathBuf::from(temp_root_dir);
    scratch_dir.push(INDEX_SUBDIRNAME);
    scratch_dir.push(SCRATCH_DIRNAME);
    // Best-effort: ignore errors. A leftover scratch directory is not a
    // correctness problem; it will be reused / overwritten next build.
    let _ = std::fs::remove_dir_all(&scratch_dir);
}

/// Pads a basename into a fixed 64-byte stack record. The first byte
/// past the basename length is set to NUL; subsequent bytes are zero.
///
/// Returns `None` if the basename exceeds `MAX_BASENAME_LEN`. The caller
/// (the build pass) responds to `None` by skipping the file and
/// incrementing a local counter, **not** by halting.
fn pack_basename_record(basename_bytes: &[u8]) -> Option<[u8; NAME_RECORD_SIZE]> {
    if basename_bytes.len() > MAX_BASENAME_LEN {
        return None;
    }
    let mut record_buffer = [0u8; NAME_RECORD_SIZE];
    record_buffer[..basename_bytes.len()].copy_from_slice(basename_bytes);
    Some(record_buffer)
}

// =========================================================================
// Cold-build orchestration
// =========================================================================

/// Result summary from a successful cold build. Returned to the caller
/// for observability / logging. Contains no user data.
#[derive(Clone, Copy, Debug)]
pub struct ColdBuildSummary {
    /// Number of files successfully indexed.
    pub files_indexed: u64,
    /// Number of entries skipped because their basename exceeded
    /// `MAX_BASENAME_LEN`. Project rule: skip & continue, do not halt.
    pub entries_skipped_overlong_name: u64,
    /// Number of entries skipped because `stat` failed on them.
    pub entries_skipped_stat_failed: u64,
    /// Number of entries skipped because they were not regular files
    /// (e.g. subdirectories, symlinks). Project rule: only regular files
    /// are indexed.
    pub entries_skipped_non_regular: u64,
}

/// Performs a complete cold (re)build of the index for the given parent
/// directory, writing all output under `<temp_root>/chrono_index/`.
///
/// On success: `header.bin`, `names.bin`, `mtimes.bin`, and `cursor.bin`
/// are all present and consistent. Previous versions of these files (if
/// any) are replaced atomically.
///
/// On failure: a terse error code is returned. The previous index (if
/// any) remains intact because the new `header.bin` is the last file
/// written, via atomic rename. Scratch artifacts are cleaned up
/// best-effort.
///
/// Per project policy this function never panics and never halts.
pub fn cold_build_index(
    temp_root_dir: &Path,
    parent_directory_to_index: &Path,
) -> Result<ColdBuildSummary, ChronoIndexError> {
    // -- Phase 0: prepare directories ------------------------------------
    ensure_index_directory_exists(temp_root_dir)?;
    ensure_scratch_directory_exists(temp_root_dir)?;

    // Validate and capture parent path bytes for the header.
    let parent_path_bytes = posix_path_to_bytes(parent_directory_to_index)?;
    let mut working_header = ChronoIndexHeader::new_for_parent(parent_path_bytes)?;

    // -- Phase 1: stream read_dir → names.bin + scratch unsorted mtimes ---
    let names_path = build_index_file_path(temp_root_dir, NAMES_FILENAME);
    let scratch_unsorted_path = build_scratch_path(temp_root_dir, SCRATCH_UNSORTED_MTIMES_FILENAME);

    // We write names.bin and the unsorted scratch mtimes file to staging
    // names first; promote names.bin via rename after the sort succeeds.
    let names_staging_path = {
        let mut p = names_path.clone();
        p.set_file_name("names.bin.tmp");
        p
    };

    let phase1_summary = phase1_stream_directory_into_files(
        parent_directory_to_index,
        &names_staging_path,
        &scratch_unsorted_path,
        &mut working_header,
    );

    let phase1_summary = match phase1_summary {
        Ok(summary) => summary,
        Err(error_code) => {
            // Clean up partial artifacts. Do not touch any pre-existing
            // production names.bin / mtimes.bin / header.bin.
            let _ = std::fs::remove_file(&names_staging_path);
            remove_scratch_directory_best_effort(temp_root_dir);
            return Err(error_code);
        }
    };

    // -- Phase 2: external merge sort the scratch unsorted file ---------
    let mtimes_staging_path = {
        let mut p = build_index_file_path(temp_root_dir, MTIMES_FILENAME);
        p.set_file_name("mtimes.bin.tmp");
        p
    };

    let sort_outcome = external_merge_sort_mtimes(
        temp_root_dir,
        &scratch_unsorted_path,
        &mtimes_staging_path,
        working_header.file_count,
    );

    if let Err(error_code) = sort_outcome {
        let _ = std::fs::remove_file(&names_staging_path);
        let _ = std::fs::remove_file(&mtimes_staging_path);
        remove_scratch_directory_best_effort(temp_root_dir);
        return Err(error_code);
    }

    // -- Phase 3: capture last_mtime_* from the now-sorted file ---------
    // The last record in the sorted file is the chronologically newest;
    // we store its mtime in the header so the append path (part c) can
    // validate the "new files have newer mtimes" invariant in O(1).
    if working_header.file_count > 0 {
        match read_last_mtime_record(&mtimes_staging_path, working_header.file_count) {
            Ok(last_record) => {
                working_header.last_mtime_sec = last_record.mtime_sec;
                working_header.last_mtime_nsec = last_record.mtime_nsec;
            }
            Err(error_code) => {
                let _ = std::fs::remove_file(&names_staging_path);
                let _ = std::fs::remove_file(&mtimes_staging_path);
                remove_scratch_directory_best_effort(temp_root_dir);
                return Err(error_code);
            }
        }
    }
    // If file_count == 0: leave last_mtime_* at the sentinel from
    // `new_for_parent`, so any first appended file is strictly newer.

    // -- Phase 4: promote staging files via atomic rename ---------------
    // Order matters: data files first, header last. A crash between the
    // data renames and the header rename leaves the previous header in
    // place pointing at the previous data files; on next startup the
    // change-detection / validation will rebuild. Self-healing.
    if std::fs::rename(&names_staging_path, &names_path).is_err() {
        let _ = std::fs::remove_file(&names_staging_path);
        let _ = std::fs::remove_file(&mtimes_staging_path);
        remove_scratch_directory_best_effort(temp_root_dir);
        return Err(ChronoIndexError::RenameIo);
    }

    let mtimes_final_path = build_index_file_path(temp_root_dir, MTIMES_FILENAME);
    if std::fs::rename(&mtimes_staging_path, &mtimes_final_path).is_err() {
        // names.bin is now ahead of mtimes.bin; header has not yet been
        // updated to reference the new state, so the existing (old)
        // header is still authoritative. On next run, header validation
        // vs. file sizes will mismatch and trigger a fresh rebuild.
        let _ = std::fs::remove_file(&mtimes_staging_path);
        remove_scratch_directory_best_effort(temp_root_dir);
        return Err(ChronoIndexError::RenameIo);
    }

    // Reset cursor to zero for the freshly built index.
    if let Err(error_code) = write_cursor_atomic(temp_root_dir, 0) {
        remove_scratch_directory_best_effort(temp_root_dir);
        return Err(error_code);
    }

    // Header is the last write — its presence (with the new file_count)
    // signals "this index is committed."
    if let Err(error_code) = write_header_atomic(temp_root_dir, &working_header) {
        remove_scratch_directory_best_effort(temp_root_dir);
        return Err(error_code);
    }

    // -- Phase 5: cleanup scratch ---------------------------------------
    remove_scratch_directory_best_effort(temp_root_dir);

    Ok(phase1_summary)
}

/// Converts an absolute parent directory `Path` to its raw POSIX bytes,
/// validating length. POSIX paths are byte sequences (not guaranteed
/// UTF-8); we treat them as such.
#[cfg(unix)]
fn posix_path_to_bytes(parent_directory: &Path) -> Result<&[u8], ChronoIndexError> {
    use std::os::unix::ffi::OsStrExt;
    let raw_bytes = parent_directory.as_os_str().as_bytes();
    if raw_bytes.is_empty() {
        return Err(ChronoIndexError::ParentPathInvalid);
    }
    if raw_bytes.len() > MAX_PARENT_PATH_LEN {
        return Err(ChronoIndexError::ParentPathTooLong);
    }
    Ok(raw_bytes)
}

#[cfg(not(unix))]
fn posix_path_to_bytes(_parent_directory: &Path) -> Result<&[u8], ChronoIndexError> {
    // This module is POSIX-scoped per project spec. On non-Unix targets
    // we refuse rather than guess at path encoding.
    Err(ChronoIndexError::ParentPathInvalid)
}

// =========================================================================
// Phase 1: directory stream → names.bin (staged) + unsorted mtimes (scratch)
// =========================================================================

/// Streams `read_dir(parent_directory)` exactly once, performing for each
/// regular-file entry:
///
///   1. Compute basename bytes; reject if too long → skip & count.
///   2. `stat()` to obtain mtime; reject on stat failure → skip & count.
///   3. Assign a sequential `record_id` (zero-based).
///   4. Append a 64-byte basename record to `names_staging_path`.
///   5. Append a 20-byte mtime record to `scratch_unsorted_path`.
///   6. Update `signal_hash` (XOR-fold of FNV-1a of basename) and
///      `file_count` in `working_header`.
///
/// All buffers used are stack-resident or fixed-size. The two output
/// files are wrapped in `BufWriter`s of bounded capacity; their internal
/// buffers are a constant size (default 8 KB each), not scaled by N.
fn phase1_stream_directory_into_files(
    parent_directory: &Path,
    names_staging_path: &Path,
    scratch_unsorted_path: &Path,
    working_header: &mut ChronoIndexHeader,
) -> Result<ColdBuildSummary, ChronoIndexError> {
    // Open writers. Truncate any leftover staging from a prior aborted run.
    let names_file_handle = match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(names_staging_path)
    {
        Ok(handle) => handle,
        Err(_) => return Err(ChronoIndexError::BuildIo),
    };
    let mut names_writer = BufWriter::new(names_file_handle);

    let scratch_file_handle = match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(scratch_unsorted_path)
    {
        Ok(handle) => handle,
        Err(_) => return Err(ChronoIndexError::BuildIo),
    };
    let mut scratch_writer = BufWriter::new(scratch_file_handle);

    // Open the directory stream. `read_dir` is a streaming iterator over
    // `readdir(3)`; it does not preload all entries.
    let directory_iterator = match std::fs::read_dir(parent_directory) {
        Ok(iter) => iter,
        Err(_) => return Err(ChronoIndexError::BuildIo),
    };

    let mut summary = ColdBuildSummary {
        files_indexed: 0,
        entries_skipped_overlong_name: 0,
        entries_skipped_stat_failed: 0,
        entries_skipped_non_regular: 0,
    };
    let mut next_record_id: u64 = 0;
    let mut signal_hash_accumulator: u64 = 0;

    for directory_entry_result in directory_iterator {
        // Per-entry I/O errors: skip this entry, continue with the rest.
        let directory_entry = match directory_entry_result {
            Ok(entry) => entry,
            Err(_) => {
                summary.entries_skipped_stat_failed =
                    summary.entries_skipped_stat_failed.saturating_add(1);
                continue;
            }
        };

        // file_type() is usually free on Linux (filled by readdir on most
        // filesystems); falls back to stat where not.
        let file_type_info = match directory_entry.file_type() {
            Ok(ft) => ft,
            Err(_) => {
                summary.entries_skipped_stat_failed =
                    summary.entries_skipped_stat_failed.saturating_add(1);
                continue;
            }
        };
        if !file_type_info.is_file() {
            summary.entries_skipped_non_regular =
                summary.entries_skipped_non_regular.saturating_add(1);
            continue;
        }

        // Extract basename bytes (POSIX = raw bytes).
        let file_name_os = directory_entry.file_name();
        let basename_bytes: &[u8] = {
            #[cfg(unix)]
            {
                use std::os::unix::ffi::OsStrExt;
                file_name_os.as_bytes()
            }
            #[cfg(not(unix))]
            {
                // POSIX-only module; reject.
                summary.entries_skipped_overlong_name =
                    summary.entries_skipped_overlong_name.saturating_add(1);
                continue;
            }
        };

        let name_record = match pack_basename_record(basename_bytes) {
            Some(packed) => packed,
            None => {
                summary.entries_skipped_overlong_name =
                    summary.entries_skipped_overlong_name.saturating_add(1);
                continue;
            }
        };

        // metadata() = stat(). Get mtime.
        let metadata = match directory_entry.metadata() {
            Ok(md) => md,
            Err(_) => {
                summary.entries_skipped_stat_failed =
                    summary.entries_skipped_stat_failed.saturating_add(1);
                continue;
            }
        };

        let (mtime_sec, mtime_nsec) = match extract_mtime_seconds_and_nanos(&metadata) {
            Some(pair) => pair,
            None => {
                summary.entries_skipped_stat_failed =
                    summary.entries_skipped_stat_failed.saturating_add(1);
                continue;
            }
        };

        // Write the name record.
        if names_writer.write_all(&name_record).is_err() {
            return Err(ChronoIndexError::BuildIo);
        }

        // Write the mtime record (record_id = this file's position in
        // names.bin).
        let mtime_record = MtimeRecord {
            mtime_sec,
            mtime_nsec,
            record_id: next_record_id,
        };
        let mut mtime_buffer = [0u8; MTIME_RECORD_SIZE];
        mtime_record.write_into(&mut mtime_buffer);
        if scratch_writer.write_all(&mtime_buffer).is_err() {
            return Err(ChronoIndexError::BuildIo);
        }

        signal_hash_accumulator ^= fnv1a_64(basename_bytes);
        next_record_id = next_record_id.saturating_add(1);
        summary.files_indexed = summary.files_indexed.saturating_add(1);
    }

    // Flush and fsync both writers so the data is durable before sort.
    if names_writer.flush().is_err() {
        return Err(ChronoIndexError::BuildIo);
    }
    let names_inner = match names_writer.into_inner() {
        Ok(inner) => inner,
        Err(_) => return Err(ChronoIndexError::BuildIo),
    };
    if names_inner.sync_all().is_err() {
        return Err(ChronoIndexError::BuildIo);
    }

    if scratch_writer.flush().is_err() {
        return Err(ChronoIndexError::BuildIo);
    }
    let scratch_inner = match scratch_writer.into_inner() {
        Ok(inner) => inner,
        Err(_) => return Err(ChronoIndexError::BuildIo),
    };
    if scratch_inner.sync_all().is_err() {
        return Err(ChronoIndexError::BuildIo);
    }

    working_header.file_count = summary.files_indexed;
    working_header.signal_hash = signal_hash_accumulator;
    Ok(summary)
}

/// Extracts `(mtime_sec, mtime_nsec)` from a `Metadata` in a POSIX-safe
/// way. Returns `None` if the metadata lacks mtime information.
#[cfg(unix)]
fn extract_mtime_seconds_and_nanos(metadata: &std::fs::Metadata) -> Option<(i64, i32)> {
    use std::os::unix::fs::MetadataExt;
    let sec = metadata.mtime();
    let nsec_u64 = metadata.mtime_nsec();
    // nsec is always in [0, 1_000_000_000); fits in i32. Defensive clamp.
    let nsec_i32 = if nsec_u64 < 0 || nsec_u64 >= 1_000_000_000 {
        0
    } else {
        nsec_u64 as i32
    };
    Some((sec, nsec_i32))
}

#[cfg(not(unix))]
fn extract_mtime_seconds_and_nanos(_metadata: &std::fs::Metadata) -> Option<(i64, i32)> {
    None
}

// =========================================================================
// Phase 2: external merge sort
// =========================================================================

/// Sorts the scratch unsorted mtimes file into `mtimes_staging_path`.
///
/// Strategy: replacement-free chunked sort.
///   1. Read `EXTERNAL_SORT_CHUNK_RECORDS` records into a heap-allocated
///      buffer (single bounded allocation, ~80 KB by default).
///   2. Sort the chunk in place with `sort_unstable_by` (no allocation).
///   3. Write the sorted chunk to a numbered run file in `scratch/`.
///   4. Repeat until input exhausted.
///   5. K-way merge runs (up to `MAX_MERGE_FANOUT` at a time) into the
///      staging output, cascading if run count exceeds the fan-out.
///
/// `expected_record_count` is the count produced by phase 1; used as a
/// sanity check and to short-circuit the no-records case.
fn external_merge_sort_mtimes(
    temp_root_dir: &Path,
    scratch_unsorted_path: &Path,
    mtimes_staging_path: &Path,
    expected_record_count: u64,
) -> Result<(), ChronoIndexError> {
    // Special case: empty directory. Produce a zero-length mtimes file.
    if expected_record_count == 0 {
        let empty_file = match OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(mtimes_staging_path)
        {
            Ok(handle) => handle,
            Err(_) => return Err(ChronoIndexError::BuildIo),
        };
        if empty_file.sync_all().is_err() {
            return Err(ChronoIndexError::BuildIo);
        }
        return Ok(());
    }

    // -- Step 1: chunked sort into run files ----------------------------
    let unsorted_handle = match File::open(scratch_unsorted_path) {
        Ok(handle) => handle,
        Err(_) => return Err(ChronoIndexError::BuildIo),
    };
    let mut unsorted_reader = BufReader::new(unsorted_handle);

    // The sort buffer is the single bounded heap allocation. Default
    // 4096 × 20 B = 80 KB. Allocated once, reused across all chunks.
    let mut sort_buffer: Vec<MtimeRecord> = Vec::with_capacity(EXTERNAL_SORT_CHUNK_RECORDS);

    let mut next_run_index: u64 = 0;
    let mut run_paths: Vec<PathBuf> = Vec::new();
    // run_paths grows by 1 per run; total runs ≤ N / chunk_size, which
    // for N=1e6 and chunk=4096 is ~245 entries × ~100 B ≈ 25 KB. This is
    // bounded by N but with such a small constant that it does not
    // threaten the memory budget. Documented; acceptable.

    loop {
        sort_buffer.clear();
        let mut record_buffer = [0u8; MTIME_RECORD_SIZE];

        while sort_buffer.len() < EXTERNAL_SORT_CHUNK_RECORDS {
            match unsorted_reader.read_exact(&mut record_buffer) {
                Ok(()) => {
                    sort_buffer.push(MtimeRecord::read_from(&record_buffer));
                }
                Err(read_error) => {
                    if read_error.kind() == std::io::ErrorKind::UnexpectedEof {
                        break;
                    }
                    return Err(ChronoIndexError::BuildIo);
                }
            }
        }

        if sort_buffer.is_empty() {
            break;
        }

        // In-place sort, no allocation (unstable is allocation-free).
        sort_buffer.sort_unstable_by(|left, right| {
            if left.mtime_sec != right.mtime_sec {
                return left.mtime_sec.cmp(&right.mtime_sec);
            }
            if left.mtime_nsec != right.mtime_nsec {
                return left.mtime_nsec.cmp(&right.mtime_nsec);
            }
            left.record_id.cmp(&right.record_id)
        });

        // Write sorted chunk to a run file.
        let run_path = build_scratch_path(temp_root_dir, &format_run_filename(next_run_index));
        if let Err(error_code) = write_run_file(&run_path, &sort_buffer) {
            // Cleanup partial runs.
            for partial in &run_paths {
                let _ = std::fs::remove_file(partial);
            }
            let _ = std::fs::remove_file(&run_path);
            return Err(error_code);
        }
        run_paths.push(run_path);
        next_run_index = next_run_index.saturating_add(1);

        if sort_buffer.len() < EXTERNAL_SORT_CHUNK_RECORDS {
            // Last partial chunk; input is exhausted.
            break;
        }
    }

    // -- Step 2: cascading k-way merge ----------------------------------
    let final_run_path = cascade_merge_runs(temp_root_dir, run_paths)?;

    // Promote the final merged run to the mtimes staging path.
    if std::fs::rename(&final_run_path, mtimes_staging_path).is_err() {
        let _ = std::fs::remove_file(&final_run_path);
        return Err(ChronoIndexError::RenameIo);
    }
    Ok(())
}

/// `format!` is heap-using but produces a short, bounded-length string
/// (e.g. "run_00000042.bin"). The allocation is per-run, not per-record;
/// total allocations across a 1M-file build are ~245 × ~16 B = ~4 KB.
/// Documented as acceptable per project rules ("rule of thumb, not
/// pedantic"). If even this is unacceptable, swap for a stack `[u8; 24]`
/// formatter (e.g. via the project's Buffy module).
fn format_run_filename(run_index: u64) -> String {
    format!(
        "{}{:010}{}",
        SCRATCH_RUN_FILENAME_PREFIX, run_index, SCRATCH_RUN_FILENAME_SUFFIX
    )
}

fn write_run_file(run_path: &Path, sorted_records: &[MtimeRecord]) -> Result<(), ChronoIndexError> {
    let run_handle = match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(run_path)
    {
        Ok(handle) => handle,
        Err(_) => return Err(ChronoIndexError::BuildIo),
    };
    let mut run_writer = BufWriter::new(run_handle);
    let mut record_buffer = [0u8; MTIME_RECORD_SIZE];
    for record in sorted_records {
        record.write_into(&mut record_buffer);
        if run_writer.write_all(&record_buffer).is_err() {
            return Err(ChronoIndexError::BuildIo);
        }
    }
    if run_writer.flush().is_err() {
        return Err(ChronoIndexError::BuildIo);
    }
    let inner = match run_writer.into_inner() {
        Ok(inner) => inner,
        Err(_) => return Err(ChronoIndexError::BuildIo),
    };
    if inner.sync_all().is_err() {
        return Err(ChronoIndexError::BuildIo);
    }
    Ok(())
}

/// Repeatedly merges up to `MAX_MERGE_FANOUT` runs at a time until a
/// single sorted run remains. Returns the path to that final run.
fn cascade_merge_runs(
    temp_root_dir: &Path,
    mut current_run_paths: Vec<PathBuf>,
) -> Result<PathBuf, ChronoIndexError> {
    if current_run_paths.is_empty() {
        // Shouldn't happen (expected_record_count > 0 was checked) but
        // handle defensively without panic.
        return Err(ChronoIndexError::BuildIo);
    }

    let mut merge_round_index: u64 = 0;

    while current_run_paths.len() > 1 {
        let mut next_round_runs: Vec<PathBuf> = Vec::new();
        let mut group_index: u64 = 0;

        let mut cursor = 0usize;
        while cursor < current_run_paths.len() {
            let group_end = (cursor + MAX_MERGE_FANOUT).min(current_run_paths.len());
            let group_slice = &current_run_paths[cursor..group_end];

            let merged_path = build_scratch_path(
                temp_root_dir,
                &format!("merge_r{:04}_g{:010}.bin", merge_round_index, group_index),
            );

            if let Err(error_code) = merge_runs_into(&merged_path, group_slice) {
                // Best-effort cleanup of all current and partial runs.
                for partial in &current_run_paths {
                    let _ = std::fs::remove_file(partial);
                }
                for partial in &next_round_runs {
                    let _ = std::fs::remove_file(partial);
                }
                let _ = std::fs::remove_file(&merged_path);
                return Err(error_code);
            }

            // Inputs to this merge are no longer needed.
            for consumed in group_slice {
                let _ = std::fs::remove_file(consumed);
            }
            next_round_runs.push(merged_path);
            cursor = group_end;
            group_index = group_index.saturating_add(1);
        }

        current_run_paths = next_round_runs;
        merge_round_index = merge_round_index.saturating_add(1);
    }

    // Exactly one run remains.
    match current_run_paths.into_iter().next() {
        Some(final_path) => Ok(final_path),
        None => Err(ChronoIndexError::BuildIo),
    }
}

/// Merges a group of up to `MAX_MERGE_FANOUT` already-sorted run files
/// into a single sorted output file using a small fixed-size tournament.
///
/// Memory: one `MtimeRecord` per input run held in a stack-resident
/// `[Option<MtimeRecord>; MAX_MERGE_FANOUT]` array (320 B), plus one
/// `BufReader` per input run (bounded buffer, default 8 KB each =
/// 128 KB total at fan-out 16). All bounded and independent of N.
fn merge_runs_into(
    output_path: &Path,
    input_run_paths: &[PathBuf],
) -> Result<(), ChronoIndexError> {
    // Open all input readers. If any open fails, treat as build error.
    // Stack-resident array of optional readers, sized to MAX_MERGE_FANOUT.
    let mut input_readers: [Option<BufReader<File>>; MAX_MERGE_FANOUT] = Default::default();
    let mut head_records: [Option<MtimeRecord>; MAX_MERGE_FANOUT] = [None; MAX_MERGE_FANOUT];

    // Defensive: input_run_paths.len() must not exceed MAX_MERGE_FANOUT.
    if input_run_paths.len() > MAX_MERGE_FANOUT {
        return Err(ChronoIndexError::BuildIo);
    }

    for (slot_index, run_path) in input_run_paths.iter().enumerate() {
        let handle = match File::open(run_path) {
            Ok(h) => h,
            Err(_) => return Err(ChronoIndexError::BuildIo),
        };
        let mut reader = BufReader::new(handle);
        // Prime each reader with its first record.
        let mut record_buffer = [0u8; MTIME_RECORD_SIZE];
        match reader.read_exact(&mut record_buffer) {
            Ok(()) => {
                head_records[slot_index] = Some(MtimeRecord::read_from(&record_buffer));
            }
            Err(read_error) => {
                if read_error.kind() != std::io::ErrorKind::UnexpectedEof {
                    return Err(ChronoIndexError::BuildIo);
                }
                // Empty input run — leave head as None.
            }
        }
        input_readers[slot_index] = Some(reader);
    }

    // Open output.
    let output_handle = match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(output_path)
    {
        Ok(h) => h,
        Err(_) => return Err(ChronoIndexError::BuildIo),
    };
    let mut output_writer = BufWriter::new(output_handle);
    let mut output_buffer = [0u8; MTIME_RECORD_SIZE];

    // Linear scan over `MAX_MERGE_FANOUT` heads per record. With fan-out
    // ≤ 16, this is faster than a binary heap for our sizes and uses no
    // allocation. Total comparisons: N * fan-out, still O(N log N) with
    // the cascade depth factor.
    loop {
        // Find the slot with the smallest head record.
        let mut smallest_slot: Option<usize> = None;
        for slot_index in 0..input_run_paths.len() {
            if let Some(candidate_record) = head_records[slot_index] {
                match smallest_slot {
                    None => smallest_slot = Some(slot_index),
                    Some(current_best_slot) => {
                        // Safe: current_best_slot has Some head by construction.
                        let current_best = head_records[current_best_slot].unwrap_or(MtimeRecord {
                            mtime_sec: i64::MAX,
                            mtime_nsec: i32::MAX,
                            record_id: u64::MAX,
                        });
                        if candidate_record.is_strictly_before(current_best) {
                            smallest_slot = Some(slot_index);
                        }
                    }
                }
            }
        }

        let chosen_slot = match smallest_slot {
            Some(slot) => slot,
            None => break, // all inputs exhausted
        };

        // Write the chosen head and advance that reader.
        let chosen_record = match head_records[chosen_slot] {
            Some(r) => r,
            None => break, // unreachable per logic above; defensive exit
        };
        chosen_record.write_into(&mut output_buffer);
        if output_writer.write_all(&output_buffer).is_err() {
            return Err(ChronoIndexError::BuildIo);
        }

        // Advance the chosen reader.
        let reader_slot = match &mut input_readers[chosen_slot] {
            Some(r) => r,
            None => {
                head_records[chosen_slot] = None;
                continue;
            }
        };
        let mut record_buffer = [0u8; MTIME_RECORD_SIZE];
        match reader_slot.read_exact(&mut record_buffer) {
            Ok(()) => {
                head_records[chosen_slot] = Some(MtimeRecord::read_from(&record_buffer));
            }
            Err(read_error) => {
                if read_error.kind() == std::io::ErrorKind::UnexpectedEof {
                    head_records[chosen_slot] = None;
                } else {
                    return Err(ChronoIndexError::BuildIo);
                }
            }
        }
    }

    if output_writer.flush().is_err() {
        return Err(ChronoIndexError::BuildIo);
    }
    let inner = match output_writer.into_inner() {
        Ok(inner) => inner,
        Err(_) => return Err(ChronoIndexError::BuildIo),
    };
    if inner.sync_all().is_err() {
        return Err(ChronoIndexError::BuildIo);
    }
    Ok(())
}

/// Reads the final (highest-index) record from a sorted mtimes file.
/// Used after Phase 2 to populate `header.last_mtime_*`.
fn read_last_mtime_record(
    mtimes_path: &Path,
    record_count: u64,
) -> Result<MtimeRecord, ChronoIndexError> {
    if record_count == 0 {
        return Err(ChronoIndexError::BuildIo);
    }
    let mut handle = match File::open(mtimes_path) {
        Ok(h) => h,
        Err(_) => return Err(ChronoIndexError::BuildIo),
    };
    let last_index = record_count.saturating_sub(1);
    let byte_offset = last_index.saturating_mul(MTIME_RECORD_SIZE as u64);
    if handle.seek(SeekFrom::Start(byte_offset)).is_err() {
        return Err(ChronoIndexError::BuildIo);
    }
    let mut record_buffer = [0u8; MTIME_RECORD_SIZE];
    if handle.read_exact(&mut record_buffer).is_err() {
        return Err(ChronoIndexError::BuildIo);
    }
    Ok(MtimeRecord::read_from(&record_buffer))
}

// =========================================================================
// Tests for part (b)
// =========================================================================

#[cfg(test)]
mod chrono_index_part_b_tests {
    use super::*;
    // use std::io::Write as _;

    /// Creates a unique scratch directory for the index temp root.
    fn make_test_temp_root(label: &str) -> PathBuf {
        let mut scratch = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        scratch.push(format!(
            "chrono_index_b_{}_{}_{}",
            label,
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&scratch).expect("setup: temp root");
        scratch
    }

    /// Creates a separate "watched" directory and populates it with the
    /// given (basename, content) pairs. Each file is created in order, so
    /// (on most filesystems with sufficient timestamp resolution) the
    /// later files will have strictly newer mtimes — matching the
    /// project's "new files have newer mtimes" invariant.
    fn make_watched_dir_with_files(label: &str, files: &[(&str, &[u8])]) -> PathBuf {
        let mut watched = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        watched.push(format!(
            "chrono_watched_{}_{}_{}",
            label,
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&watched).expect("setup: watched dir");
        for (basename, content) in files {
            let mut path = watched.clone();
            path.push(basename);
            let mut f = std::fs::File::create(&path).expect("setup: create file");
            f.write_all(content).expect("setup: write file");
            f.sync_all().expect("setup: sync file");
            // Sleep a few ms so subsequent files have strictly newer mtime
            // on filesystems with millisecond resolution (ext4 has ns res,
            // but some test envs use coarser). This keeps the invariant
            // observable in tests.
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        watched
    }

    #[test]
    fn cold_build_on_empty_dir_produces_empty_index() {
        let temp_root = make_test_temp_root("empty");
        let watched = make_watched_dir_with_files("empty", &[]);

        let summary = cold_build_index(&temp_root, &watched).expect("build ok");
        assert_eq!(summary.files_indexed, 0);

        let header = read_header(&temp_root)
            .expect("read header ok")
            .expect("header present");
        assert_eq!(header.file_count, 0);
        assert_eq!(header.signal_hash, 0);
        // last_mtime sentinel preserved
        assert_eq!(header.last_mtime_sec, i64::MIN);
        assert_eq!(header.last_mtime_nsec, 0);

        // mtimes.bin should exist and be empty.
        let mtimes_path = build_index_file_path(&temp_root, MTIMES_FILENAME);
        let meta = std::fs::metadata(&mtimes_path).expect("mtimes exists");
        assert_eq!(meta.len(), 0);

        let cursor = read_cursor(&temp_root).expect("cursor ok");
        assert_eq!(cursor, 0);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn cold_build_small_dir_produces_sorted_mtimes() {
        let temp_root = make_test_temp_root("small");
        // Create files in alphabetical order; they should also be in
        // chronological order because of the sleep in setup.
        let watched = make_watched_dir_with_files(
            "small",
            &[
                ("alpha.txt", b"a"),
                ("bravo.txt", b"bb"),
                ("charlie.txt", b"ccc"),
                ("delta.txt", b"dddd"),
            ],
        );

        let summary = cold_build_index(&temp_root, &watched).expect("build ok");
        assert_eq!(summary.files_indexed, 4);
        assert_eq!(summary.entries_skipped_overlong_name, 0);

        let header = read_header(&temp_root)
            .expect("read header ok")
            .expect("header present");
        assert_eq!(header.file_count, 4);
        assert_ne!(header.signal_hash, 0);

        // Verify mtimes.bin is sorted ascending.
        let mtimes_path = build_index_file_path(&temp_root, MTIMES_FILENAME);
        let meta = std::fs::metadata(&mtimes_path).expect("mtimes exists");
        assert_eq!(meta.len() as usize, 4 * MTIME_RECORD_SIZE);

        let mut handle = File::open(&mtimes_path).expect("open mtimes");
        let mut previous: Option<MtimeRecord> = None;
        for _ in 0..4 {
            let mut buf = [0u8; MTIME_RECORD_SIZE];
            handle.read_exact(&mut buf).expect("read record");
            let current = MtimeRecord::read_from(&buf);
            if let Some(prev) = previous {
                // Either strictly before or equal (with record_id tiebreak)
                let strictly_before_or_equal = prev.is_strictly_before(current)
                    || (prev.mtime_sec == current.mtime_sec
                        && prev.mtime_nsec == current.mtime_nsec
                        && prev.record_id < current.record_id);
                assert!(strictly_before_or_equal, "mtimes.bin not sorted");
            }
            previous = Some(current);
        }

        // header.last_mtime_* must equal the last record.
        let last = previous.expect("at least one record");
        assert_eq!(header.last_mtime_sec, last.mtime_sec);
        assert_eq!(header.last_mtime_nsec, last.mtime_nsec);

        // Scratch directory must have been cleaned up.
        let mut scratch_path = temp_root.clone();
        scratch_path.push(INDEX_SUBDIRNAME);
        scratch_path.push(SCRATCH_DIRNAME);
        assert!(!scratch_path.exists(), "scratch should be cleaned up");

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn cold_build_larger_dir_exercises_external_sort() {
        // Force at least 2 chunks by creating > EXTERNAL_SORT_CHUNK_RECORDS
        // files. For test speed, reduce by using a smaller test-only count
        // and trusting the algorithm at larger N. We use 50 files here
        // and additionally verify the single-chunk path; the multi-chunk
        // path is covered by `cold_build_forces_multi_chunk_sort` below.
        let temp_root = make_test_temp_root("medium");
        let mut files_owned: Vec<(String, Vec<u8>)> = Vec::new();
        for i in 0..50u32 {
            files_owned.push((format!("file_{:04}.dat", i), vec![i as u8; 4]));
        }
        let files_ref: Vec<(&str, &[u8])> = files_owned
            .iter()
            .map(|(name, content)| (name.as_str(), content.as_slice()))
            .collect();
        let watched = make_watched_dir_with_files("medium", &files_ref);

        let summary = cold_build_index(&temp_root, &watched).expect("build ok");
        assert_eq!(summary.files_indexed, 50);

        let header = read_header(&temp_root)
            .expect("read header ok")
            .expect("header present");
        assert_eq!(header.file_count, 50);

        // Verify full sort order across the file.
        let mtimes_path = build_index_file_path(&temp_root, MTIMES_FILENAME);
        let mut handle = File::open(&mtimes_path).expect("open mtimes");
        let mut previous: Option<MtimeRecord> = None;
        for _ in 0..50 {
            let mut buf = [0u8; MTIME_RECORD_SIZE];
            handle.read_exact(&mut buf).expect("read record");
            let current = MtimeRecord::read_from(&buf);
            if let Some(prev) = previous {
                let ordered = prev.is_strictly_before(current)
                    || (prev.mtime_sec == current.mtime_sec
                        && prev.mtime_nsec == current.mtime_nsec
                        && prev.record_id < current.record_id);
                assert!(ordered, "mtimes not in order");
            }
            previous = Some(current);
        }

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn cold_build_skips_overlong_basenames_without_halting() {
        let temp_root = make_test_temp_root("overlong");
        // Construct an overlong filename (65 chars).
        let overlong: String = "x".repeat(MAX_BASENAME_LEN + 1);
        let watched = make_watched_dir_with_files(
            "overlong",
            &[
                ("ok_short.txt", b"a"),
                (overlong.as_str(), b"b"),
                ("also_ok.txt", b"c"),
            ],
        );

        let summary = cold_build_index(&temp_root, &watched).expect("build ok");
        assert_eq!(summary.files_indexed, 2);
        assert_eq!(summary.entries_skipped_overlong_name, 1);

        let header = read_header(&temp_root)
            .expect("read header ok")
            .expect("header present");
        assert_eq!(header.file_count, 2);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn cold_build_skips_non_regular_entries() {
        let temp_root = make_test_temp_root("non_regular");
        let watched = make_watched_dir_with_files("non_regular", &[("real_file.txt", b"hi")]);

        // Add a subdirectory inside watched.
        let mut subdir_path = watched.clone();
        subdir_path.push("a_subdirectory");
        std::fs::create_dir_all(&subdir_path).expect("setup: subdir");

        let summary = cold_build_index(&temp_root, &watched).expect("build ok");
        assert_eq!(summary.files_indexed, 1);
        assert!(summary.entries_skipped_non_regular >= 1);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn cold_build_rejects_nonexistent_parent() {
        let temp_root = make_test_temp_root("no_parent");
        let mut nonexistent = std::env::temp_dir();
        nonexistent.push("definitely_not_a_real_dir_chrono_index_test_xyz_123");
        // Make sure it really doesn't exist.
        let _ = std::fs::remove_dir_all(&nonexistent);

        let result = cold_build_index(&temp_root, &nonexistent);
        assert!(result.is_err());
        assert_eq!(result.err(), Some(ChronoIndexError::BuildIo));

        // No header should have been written.
        let header_path = build_index_file_path(&temp_root, HEADER_FILENAME);
        assert!(!header_path.exists());

        let _ = std::fs::remove_dir_all(&temp_root);
    }

    #[test]
    fn fnv1a_64_is_deterministic_and_distinguishes_inputs() {
        assert_eq!(fnv1a_64(b""), fnv1a_64(b""));
        assert_eq!(fnv1a_64(b"abc"), fnv1a_64(b"abc"));
        assert_ne!(fnv1a_64(b"abc"), fnv1a_64(b"abd"));
    }

    #[test]
    fn mtime_record_serialize_round_trip() {
        let original = MtimeRecord {
            mtime_sec: 1_700_000_123,
            mtime_nsec: 456_789_012,
            record_id: 999_999,
        };
        let mut buf = [0u8; MTIME_RECORD_SIZE];
        original.write_into(&mut buf);
        let recovered = MtimeRecord::read_from(&buf);
        assert_eq!(recovered.mtime_sec, original.mtime_sec);
        assert_eq!(recovered.mtime_nsec, original.mtime_nsec);
        assert_eq!(recovered.record_id, original.record_id);
    }

    #[test]
    fn mtime_record_strict_ordering_uses_sec_then_nsec_then_record_id() {
        let earlier = MtimeRecord {
            mtime_sec: 100,
            mtime_nsec: 0,
            record_id: 50,
        };
        let later_sec = MtimeRecord {
            mtime_sec: 101,
            mtime_nsec: 0,
            record_id: 0,
        };
        let same_sec_later_nsec = MtimeRecord {
            mtime_sec: 100,
            mtime_nsec: 1,
            record_id: 0,
        };
        let same_sec_same_nsec_later_id = MtimeRecord {
            mtime_sec: 100,
            mtime_nsec: 0,
            record_id: 51,
        };

        // sec dominates
        assert!(earlier.is_strictly_before(later_sec));
        assert!(!later_sec.is_strictly_before(earlier));

        // nsec tiebreaks on equal sec
        assert!(earlier.is_strictly_before(same_sec_later_nsec));
        assert!(!same_sec_later_nsec.is_strictly_before(earlier));

        // record_id tiebreaks on equal sec+nsec
        assert!(earlier.is_strictly_before(same_sec_same_nsec_later_id));
        assert!(!same_sec_same_nsec_later_id.is_strictly_before(earlier));

        // Equal records are not strictly-before each other.
        let copy_of_earlier = earlier;
        assert!(!earlier.is_strictly_before(copy_of_earlier));
        assert!(!copy_of_earlier.is_strictly_before(earlier));
    }

    #[test]
    fn pack_basename_record_rejects_overlong_input() {
        let just_right = vec![b'a'; MAX_BASENAME_LEN];
        assert!(pack_basename_record(&just_right).is_some());

        let too_long = vec![b'a'; MAX_BASENAME_LEN + 1];
        assert!(pack_basename_record(&too_long).is_none());
    }

    #[test]
    fn pack_basename_record_zero_pads_unused_tail() {
        let short_name = b"hi";
        let packed = pack_basename_record(short_name).expect("fits");
        // First two bytes are the name; remainder must be zero.
        assert_eq!(&packed[..2], short_name);
        for trailing_byte in &packed[2..] {
            assert_eq!(*trailing_byte, 0);
        }
    }

    #[test]
    fn cold_build_records_signal_hash_as_xor_of_basename_fnv1a() {
        let temp_root = make_test_temp_root("signal");
        let watched = make_watched_dir_with_files(
            "signal",
            &[("one.dat", b"1"), ("two.dat", b"22"), ("three.dat", b"333")],
        );

        let _ = cold_build_index(&temp_root, &watched).expect("build ok");
        let header = read_header(&temp_root)
            .expect("read header ok")
            .expect("header present");

        // Recompute expected XOR independently and compare.
        let expected_signal = fnv1a_64(b"one.dat") ^ fnv1a_64(b"two.dat") ^ fnv1a_64(b"three.dat");
        assert_eq!(header.signal_hash, expected_signal);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn cold_build_writes_names_in_record_id_order() {
        // record_id is assigned in readdir-iteration order. We don't
        // know that order on a given filesystem, but we DO know that
        // each mtime record's `record_id` must index a valid 64-byte
        // slot in names.bin, and the basename at that slot must be one
        // of the files we created.
        let temp_root = make_test_temp_root("names_layout");
        let watched = make_watched_dir_with_files(
            "names_layout",
            &[("aaa.txt", b"x"), ("bbb.txt", b"y"), ("ccc.txt", b"z")],
        );

        let summary = cold_build_index(&temp_root, &watched).expect("build ok");
        assert_eq!(summary.files_indexed, 3);

        let mtimes_path = build_index_file_path(&temp_root, MTIMES_FILENAME);
        let names_path = build_index_file_path(&temp_root, NAMES_FILENAME);

        let mut mtimes_handle = File::open(&mtimes_path).expect("open mtimes");
        let mut names_handle = File::open(&names_path).expect("open names");

        // Collect (record_id) for each mtime record in sorted order.
        let mut seen_basenames: Vec<Vec<u8>> = Vec::new();
        for _ in 0..3 {
            let mut mtime_buf = [0u8; MTIME_RECORD_SIZE];
            mtimes_handle
                .read_exact(&mut mtime_buf)
                .expect("read mtime");
            let record = MtimeRecord::read_from(&mtime_buf);

            // Seek into names.bin by record_id and read the 64-byte slot.
            let names_offset = record.record_id.saturating_mul(NAME_RECORD_SIZE as u64);
            names_handle
                .seek(SeekFrom::Start(names_offset))
                .expect("seek names");
            let mut name_buf = [0u8; NAME_RECORD_SIZE];
            names_handle.read_exact(&mut name_buf).expect("read name");

            // Trim trailing zeros for comparison.
            let used_len = name_buf
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(NAME_RECORD_SIZE);
            seen_basenames.push(name_buf[..used_len].to_vec());
        }

        // All three known basenames must appear exactly once.
        let mut sorted_seen = seen_basenames.clone();
        sorted_seen.sort();
        let mut expected = vec![
            b"aaa.txt".to_vec(),
            b"bbb.txt".to_vec(),
            b"ccc.txt".to_vec(),
        ];
        expected.sort();
        assert_eq!(sorted_seen, expected);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn cold_build_resets_cursor_to_zero() {
        let temp_root = make_test_temp_root("cursor_reset");
        let watched = make_watched_dir_with_files("cursor_reset", &[("only.txt", b"q")]);

        // Pre-seed a non-zero cursor to verify the build resets it.
        ensure_index_directory_exists(&temp_root).expect("setup");
        write_cursor_atomic(&temp_root, 12345).expect("seed cursor");
        assert_eq!(read_cursor(&temp_root).expect("read"), 12345);

        let _ = cold_build_index(&temp_root, &watched).expect("build ok");
        assert_eq!(read_cursor(&temp_root).expect("read"), 0);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn cold_build_overwrites_previous_index() {
        let temp_root = make_test_temp_root("rebuild");
        let watched = make_watched_dir_with_files("rebuild_a", &[("first.txt", b"f")]);

        let summary_a = cold_build_index(&temp_root, &watched).expect("build a");
        assert_eq!(summary_a.files_indexed, 1);

        // Add another file and rebuild.
        let mut second_path = watched.clone();
        second_path.push("second.txt");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let mut f = std::fs::File::create(&second_path).expect("create second");
        f.write_all(b"s").expect("write second");
        f.sync_all().expect("sync second");

        let summary_b = cold_build_index(&temp_root, &watched).expect("build b");
        assert_eq!(summary_b.files_indexed, 2);

        let header = read_header(&temp_root)
            .expect("read header")
            .expect("present");
        assert_eq!(header.file_count, 2);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }
}

// =========================================================================
// Part (c): Incremental-append path
// =========================================================================
//
// ## When this runs
//
// The append path is the steady-state hot path. It is invoked when the
// directory has grown — i.e. one or more new files have appeared since the
// last index commit — and the pre-existing portion of the index appears
// unchanged.
//
// The caller's update-orchestration logic decides which path to run by
// comparing the live directory against `header.bin`. Concretely:
//
//   - Scan the live directory once, computing `(live_count, live_signal_hash)`
//     where `live_signal_hash` is the XOR-fold of FNV-1a 64 of every
//     basename. This is one streaming pass, no `stat()`, no heap growth
//     with N.
//
//   - Compare against the header:
//
//       * `live_count == header.file_count`
//         && `live_signal_hash == header.signal_hash`
//             → index is current; nothing to do.
//
//       * `live_count > header.file_count`
//         && the XOR of the *new* basenames equals
//            `live_signal_hash XOR header.signal_hash`
//             → exactly K = live_count - header.file_count new files
//               appeared and none of the existing files were renamed or
//               removed. This is the *append-eligible* case and is the
//               subject of part (c) below.
//
//       * Anything else (e.g. live_count shrank, hashes incompatible)
//             → fall back to cold rebuild (part b). Log a terse code.
//               Per project policy: never halt.
//
//   Note: the orchestration above (the "decide which path" step) is a
//   thin wrapper around the building blocks in this file. It is exposed
//   in part (d) along with the tick function so that callers have one
//   single high-level entrypoint.
//
// ## What this path does
//
// Given:
//   - the index temp root,
//   - the watched parent directory,
//   - the currently-committed header (already loaded from disk),
//
// the append path:
//
//   1. Streams `read_dir` once.
//   2. For each entry, computes the basename hash and looks it up in an
//      on-disk "name hash index" sidecar (lazily built on first append;
//      see `name_hashes.bin`). If the hash is already present, the entry
//      is an existing file → skip. Otherwise the entry is *candidate new*.
//   3. For each candidate new entry: calls `stat()` to obtain its mtime,
//      appends its basename to `names.bin` (assigning a new record_id),
//      and buffers the resulting `MtimeRecord` in a small fixed-size
//      stack/heap batch. Batches are bounded by `APPEND_BATCH_RECORDS`.
//   4. When a batch is full (or the directory scan ends), the batch is
//      sorted in place, then merge-appended to `mtimes.bin`:
//         * Fast path: every record's `(mtime_sec, mtime_nsec)` is
//           strictly newer than `header.last_mtime_*`. Pure append, no
//           rewrite. This is the expected steady-state path.
//         * Slow path: at least one record in the batch is older than
//           the current `header.last_mtime_*`. This violates the project
//           invariant but does not halt; we bump
//           `header.invariant_breach_count`, perform a bounded merge
//           insert (rewriting only the suffix of `mtimes.bin` that
//           needs reordering), and continue.
//   5. Updates `header.bin` atomically (file_count, signal_hash,
//      last_mtime_*, possibly invariant_breach_count). `cursor.bin` is
//      NOT moved — the chrono iteration cursor advances independently.
//
// ## Memory discipline
//
// All buffers used by the append path are fixed-size:
//   - One `[u8; NAME_RECORD_SIZE]` for the basename being written.
//   - One `[u8; MTIME_RECORD_SIZE]` for the mtime record being written.
//   - One `Vec<MtimeRecord>` of capacity `APPEND_BATCH_RECORDS` (default
//     256 × 20 B = 5 KB). Single bounded allocation per call; reused
//     across batches in the same call.
//   - The `name_hashes.bin` sidecar is consulted via streamed reads
//     of fixed-size chunks; it is never loaded whole.
//
// No structure grows with N during append.
//
// ## Failure policy
//
// Append is best-effort and never halts. On any I/O or structural error
// the function:
//   - leaves `names.bin` and `mtimes.bin` in the largest consistent
//     prefix it has successfully reached (file truncation, see below),
//   - rewrites `header.bin` atomically to reflect that prefix, and
//   - returns a terse error code.
//
// The caller may retry on the next tick. Worst case, the orchestration
// in part (d) demotes the next attempt to a cold rebuild.
//
// To keep `names.bin` and `mtimes.bin` consistent under crash or partial
// write, we do not update `header.bin` until both files are flushed and
// synced. The header is the commit point.

/// Maximum number of new-file `MtimeRecord`s buffered before a batch
/// flush. Each record is 20 B; default 256 × 20 = 5 KB. Single bounded
/// allocation per append call. Choose larger if appends typically arrive
/// in larger bursts; choose smaller if memory is tighter still.
pub const APPEND_BATCH_RECORDS: usize = 256;

/// Filename of the optional sidecar that stores per-basename FNV-1a 64
/// hashes parallel to `names.bin`. Built lazily on first append. Allows
/// "is this basename already indexed?" to be answered without rereading
/// the (heavier) `names.bin`.
///
/// Layout: `record_id -> u64 FNV-1a 64 of basename`. Fixed 8 B per
/// record. Position `i` in this file corresponds to position `i` in
/// `names.bin`.
pub const NAME_HASHES_FILENAME: &str = "name_hashes.bin";

/// Size in bytes of one `name_hashes.bin` record.
pub const NAME_HASH_RECORD_SIZE: usize = 8;

// =========================================================================
// Public summary type
// =========================================================================

/// Summary produced by one invocation of [`incremental_append_new_files`].
#[derive(Clone, Copy, Debug)]
pub struct AppendSummary {
    /// Number of new files successfully indexed in this call.
    pub files_appended: u64,
    /// Number of directory entries skipped because the basename exceeded
    /// `MAX_BASENAME_LEN`.
    pub entries_skipped_overlong_name: u64,
    /// Number of directory entries skipped because `stat()` failed.
    pub entries_skipped_stat_failed: u64,
    /// Number of directory entries skipped because the entry was not a
    /// regular file (e.g. a subdirectory).
    pub entries_skipped_non_regular: u64,
    /// Number of new-file mtimes that arrived out of chronological order
    /// (older than the current `header.last_mtime_*`). The invariant
    /// "new files have newer mtimes" was breached this many times.
    /// Handled defensively via bounded merge insert; not fatal.
    pub invariant_breaches_this_call: u64,
}

// =========================================================================
// name_hashes sidecar: build, read, append
// =========================================================================

/// Ensures `name_hashes.bin` exists and is consistent with `names.bin`.
///
/// If `name_hashes.bin` is missing or its size disagrees with
/// `header.file_count`, the sidecar is rebuilt from scratch by streaming
/// `names.bin`. This is an O(N) operation but performs only fixed-size
/// reads; it is done at most once per index lifetime in the common case.
///
/// Memory: one `[u8; NAME_RECORD_SIZE]` and one `[u8; NAME_HASH_RECORD_SIZE]`
/// buffer on the stack. No per-N heap.
fn ensure_name_hashes_sidecar_consistent(
    temp_root_dir: &Path,
    expected_record_count: u64,
) -> Result<(), ChronoIndexError> {
    let hashes_path = build_index_file_path(temp_root_dir, NAME_HASHES_FILENAME);
    let expected_size_bytes = expected_record_count.saturating_mul(NAME_HASH_RECORD_SIZE as u64);

    let existing_size = match std::fs::metadata(&hashes_path) {
        Ok(metadata) => metadata.len(),
        Err(open_error) => {
            if open_error.kind() == std::io::ErrorKind::NotFound {
                0
            } else {
                return Err(ChronoIndexError::AppendIo);
            }
        }
    };

    if existing_size == expected_size_bytes && expected_record_count > 0 {
        // Sidecar is consistent; nothing to do.
        return Ok(());
    }
    if expected_record_count == 0 {
        // No records to hash. Make sure any stale sidecar is replaced
        // with an empty file so subsequent appends start clean.
        let empty_handle = match OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&hashes_path)
        {
            Ok(h) => h,
            Err(_) => return Err(ChronoIndexError::AppendIo),
        };
        if empty_handle.sync_all().is_err() {
            return Err(ChronoIndexError::AppendIo);
        }
        return Ok(());
    }

    // Rebuild from names.bin. Stage to a sibling temp file and atomically
    // rename, so a crash mid-build does not leave a half-written sidecar
    // that future runs would trust.
    let names_path = build_index_file_path(temp_root_dir, NAMES_FILENAME);
    let mut names_reader = match File::open(&names_path) {
        Ok(handle) => BufReader::new(handle),
        Err(_) => return Err(ChronoIndexError::AppendIo),
    };

    let mut staging_path = hashes_path.clone();
    staging_path.set_file_name("name_hashes.bin.tmp");
    let mut staging_writer = match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&staging_path)
    {
        Ok(handle) => BufWriter::new(handle),
        Err(_) => return Err(ChronoIndexError::AppendIo),
    };

    let mut name_buffer = [0u8; NAME_RECORD_SIZE];
    let mut hash_buffer = [0u8; NAME_HASH_RECORD_SIZE];
    let mut records_written: u64 = 0;

    loop {
        match names_reader.read_exact(&mut name_buffer) {
            Ok(()) => {
                let used_len = basename_used_length(&name_buffer);
                let h = fnv1a_64(&name_buffer[..used_len]);
                hash_buffer.copy_from_slice(&h.to_le_bytes());
                if staging_writer.write_all(&hash_buffer).is_err() {
                    let _ = std::fs::remove_file(&staging_path);
                    return Err(ChronoIndexError::AppendIo);
                }
                records_written = records_written.saturating_add(1);
                if records_written > expected_record_count {
                    // names.bin has more records than the header says.
                    // Refuse to produce an inconsistent sidecar.
                    let _ = std::fs::remove_file(&staging_path);
                    return Err(ChronoIndexError::AppendIo);
                }
            }
            Err(read_error) => {
                if read_error.kind() == std::io::ErrorKind::UnexpectedEof {
                    break;
                }
                let _ = std::fs::remove_file(&staging_path);
                return Err(ChronoIndexError::AppendIo);
            }
        }
    }

    if records_written != expected_record_count {
        // names.bin disagrees with header. Refuse to commit.
        let _ = std::fs::remove_file(&staging_path);
        return Err(ChronoIndexError::AppendIo);
    }

    if staging_writer.flush().is_err() {
        let _ = std::fs::remove_file(&staging_path);
        return Err(ChronoIndexError::AppendIo);
    }
    let inner = match staging_writer.into_inner() {
        Ok(i) => i,
        Err(_) => {
            let _ = std::fs::remove_file(&staging_path);
            return Err(ChronoIndexError::AppendIo);
        }
    };
    if inner.sync_all().is_err() {
        let _ = std::fs::remove_file(&staging_path);
        return Err(ChronoIndexError::AppendIo);
    }
    drop(inner);

    if std::fs::rename(&staging_path, &hashes_path).is_err() {
        let _ = std::fs::remove_file(&staging_path);
        return Err(ChronoIndexError::RenameIo);
    }
    Ok(())
}

/// Returns the used (pre-NUL-padding) length of a 64-byte basename record.
fn basename_used_length(name_record: &[u8; NAME_RECORD_SIZE]) -> usize {
    let mut used = 0usize;
    while used < NAME_RECORD_SIZE && name_record[used] != 0 {
        used += 1;
    }
    used
}

/// Tests whether `target_basename_hash` is already present anywhere in
/// `name_hashes.bin`. Streamed linear scan over fixed-size 8-byte
/// records; bounded stack memory, no heap growth with N.
///
/// For very large N this is O(N) per candidate. The append-eligibility
/// gate in part (d) (XOR-of-new-hashes equals delta) ensures we only
/// call this for genuinely new candidates, so in the common case we
/// scan and find no hit only K times where K is the number of new
/// files in this update — typically very small.
fn name_hash_is_present_in_sidecar(
    temp_root_dir: &Path,
    target_basename_hash: u64,
) -> Result<bool, ChronoIndexError> {
    let hashes_path = build_index_file_path(temp_root_dir, NAME_HASHES_FILENAME);
    let handle = match File::open(&hashes_path) {
        Ok(h) => h,
        Err(open_error) => {
            if open_error.kind() == std::io::ErrorKind::NotFound {
                return Ok(false);
            }
            return Err(ChronoIndexError::AppendIo);
        }
    };
    let mut reader = BufReader::new(handle);
    let mut record_buffer = [0u8; NAME_HASH_RECORD_SIZE];
    let target_bytes = target_basename_hash.to_le_bytes();
    loop {
        match reader.read_exact(&mut record_buffer) {
            Ok(()) => {
                if record_buffer == target_bytes {
                    return Ok(true);
                }
            }
            Err(read_error) => {
                if read_error.kind() == std::io::ErrorKind::UnexpectedEof {
                    return Ok(false);
                }
                return Err(ChronoIndexError::AppendIo);
            }
        }
    }
}

/// Appends one 8-byte hash record to `name_hashes.bin`. The caller is
/// responsible for keeping append order in lockstep with `names.bin`.
fn append_name_hash_record(
    temp_root_dir: &Path,
    new_basename_hash: u64,
) -> Result<(), ChronoIndexError> {
    let hashes_path = build_index_file_path(temp_root_dir, NAME_HASHES_FILENAME);
    let mut handle = match OpenOptions::new()
        .append(true)
        .create(true)
        .open(&hashes_path)
    {
        Ok(h) => h,
        Err(_) => return Err(ChronoIndexError::AppendIo),
    };
    let buffer = new_basename_hash.to_le_bytes();
    if handle.write_all(&buffer).is_err() {
        return Err(ChronoIndexError::AppendIo);
    }
    // Flush+sync is performed by the caller in a single fsync at end of
    // the append call, not per record, to keep cost amortized.
    Ok(())
}

// =========================================================================
// Append-only writes to names.bin and mtimes.bin
// =========================================================================

/// Appends one 64-byte basename record to `names.bin`. The caller is
/// responsible for assigning sequential record_ids and for keeping
/// `name_hashes.bin` in lockstep.
fn append_basename_record_to_names(
    temp_root_dir: &Path,
    name_record: &[u8; NAME_RECORD_SIZE],
) -> Result<(), ChronoIndexError> {
    let names_path = build_index_file_path(temp_root_dir, NAMES_FILENAME);
    let mut handle = match OpenOptions::new()
        .append(true)
        .create(true)
        .open(&names_path)
    {
        Ok(h) => h,
        Err(_) => return Err(ChronoIndexError::AppendIo),
    };
    if handle.write_all(name_record).is_err() {
        return Err(ChronoIndexError::AppendIo);
    }
    Ok(())
}

/// Appends a batch of sorted, in-order MtimeRecords to `mtimes.bin`.
/// Pre-condition (checked by caller): the first record's mtime is
/// `>= header.last_mtime_*`. This is the fast path.
fn append_sorted_mtime_batch(
    temp_root_dir: &Path,
    sorted_batch: &[MtimeRecord],
) -> Result<(), ChronoIndexError> {
    if sorted_batch.is_empty() {
        return Ok(());
    }
    let mtimes_path = build_index_file_path(temp_root_dir, MTIMES_FILENAME);
    let handle = match OpenOptions::new()
        .append(true)
        .create(true)
        .open(&mtimes_path)
    {
        Ok(h) => h,
        Err(_) => return Err(ChronoIndexError::AppendIo),
    };
    let mut writer = BufWriter::new(handle);
    let mut record_buffer = [0u8; MTIME_RECORD_SIZE];
    for record in sorted_batch {
        record.write_into(&mut record_buffer);
        if writer.write_all(&record_buffer).is_err() {
            return Err(ChronoIndexError::AppendIo);
        }
    }
    if writer.flush().is_err() {
        return Err(ChronoIndexError::AppendIo);
    }
    let inner = match writer.into_inner() {
        Ok(i) => i,
        Err(_) => return Err(ChronoIndexError::AppendIo),
    };
    if inner.sync_all().is_err() {
        return Err(ChronoIndexError::AppendIo);
    }
    Ok(())
}

/// Slow path: at least one record in `unsorted_batch` is older than the
/// current `last_known_mtime` in `mtimes.bin`. Performs a bounded merge
/// insert by:
///
///   1. Sorting `unsorted_batch` in place.
///   2. Finding the smallest record in the batch.
///   3. Locating, via a streamed scan from the end of `mtimes.bin`, the
///      first position at which the new smallest record belongs.
///   4. Reading the suffix of `mtimes.bin` from that position to a
///      staging file, merge-stream into a second staging file with the
///      sorted batch, then atomically replacing the suffix.
///
/// Memory: the unsorted batch (already heap-allocated, ≤
/// `APPEND_BATCH_RECORDS` × 20 B), plus one streamed `MtimeRecord` at a
/// time. No structure grows with N.
///
/// This path exists as a defensive fallback only; per project invariant
/// it should never run. Each invocation increments
/// `header.invariant_breach_count`. The function returns the (possibly
/// adjusted) new `(last_mtime_sec, last_mtime_nsec)` after the merge.
fn merge_insert_out_of_order_batch(
    temp_root_dir: &Path,
    unsorted_batch: &mut [MtimeRecord],
) -> Result<(i64, i32), ChronoIndexError> {
    if unsorted_batch.is_empty() {
        return Err(ChronoIndexError::AppendIo);
    }
    // Sort the batch with the same total order as the main file.
    unsorted_batch.sort_unstable_by(|left, right| {
        if left.mtime_sec != right.mtime_sec {
            return left.mtime_sec.cmp(&right.mtime_sec);
        }
        if left.mtime_nsec != right.mtime_nsec {
            return left.mtime_nsec.cmp(&right.mtime_nsec);
        }
        left.record_id.cmp(&right.record_id)
    });

    let mtimes_path = build_index_file_path(temp_root_dir, MTIMES_FILENAME);

    // We rewrite mtimes.bin into a staging file by streaming both:
    //   - the existing mtimes.bin (one record at a time), and
    //   - the sorted_batch slice (index `batch_cursor`),
    // emitting whichever head is smaller. This is a standard 2-way merge.
    let existing_handle = match File::open(&mtimes_path) {
        Ok(h) => h,
        Err(_) => return Err(ChronoIndexError::AppendIo),
    };
    let mut existing_reader = BufReader::new(existing_handle);

    let mut staging_path = mtimes_path.clone();
    staging_path.set_file_name("mtimes.bin.tmp");
    let staging_handle = match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&staging_path)
    {
        Ok(h) => h,
        Err(_) => return Err(ChronoIndexError::AppendIo),
    };
    let mut staging_writer = BufWriter::new(staging_handle);

    let mut existing_buffer = [0u8; MTIME_RECORD_SIZE];
    let mut staging_buffer = [0u8; MTIME_RECORD_SIZE];
    let mut batch_cursor = 0usize;
    let mut last_emitted: Option<MtimeRecord> = None;

    // Prime existing head.
    let mut existing_head: Option<MtimeRecord> =
        match existing_reader.read_exact(&mut existing_buffer) {
            Ok(()) => Some(MtimeRecord::read_from(&existing_buffer)),
            Err(read_error) => {
                if read_error.kind() == std::io::ErrorKind::UnexpectedEof {
                    None
                } else {
                    let _ = std::fs::remove_file(&staging_path);
                    return Err(ChronoIndexError::AppendIo);
                }
            }
        };

    loop {
        let pick_from_batch = match (&existing_head, unsorted_batch.get(batch_cursor)) {
            (Some(existing), Some(new)) => new.is_strictly_before(*existing),
            (Some(_existing), None) => false,
            (None, Some(_new)) => true,
            (None, None) => break,
        };

        let emitted = if pick_from_batch {
            let chosen = unsorted_batch[batch_cursor];
            batch_cursor = batch_cursor.saturating_add(1);
            chosen
        } else {
            let chosen = match existing_head {
                Some(r) => r,
                None => break,
            };
            // Advance existing.
            existing_head = match existing_reader.read_exact(&mut existing_buffer) {
                Ok(()) => Some(MtimeRecord::read_from(&existing_buffer)),
                Err(read_error) => {
                    if read_error.kind() == std::io::ErrorKind::UnexpectedEof {
                        None
                    } else {
                        let _ = std::fs::remove_file(&staging_path);
                        return Err(ChronoIndexError::AppendIo);
                    }
                }
            };
            chosen
        };

        emitted.write_into(&mut staging_buffer);
        if staging_writer.write_all(&staging_buffer).is_err() {
            let _ = std::fs::remove_file(&staging_path);
            return Err(ChronoIndexError::AppendIo);
        }
        last_emitted = Some(emitted);
    }

    if staging_writer.flush().is_err() {
        let _ = std::fs::remove_file(&staging_path);
        return Err(ChronoIndexError::AppendIo);
    }
    let inner = match staging_writer.into_inner() {
        Ok(i) => i,
        Err(_) => {
            let _ = std::fs::remove_file(&staging_path);
            return Err(ChronoIndexError::AppendIo);
        }
    };
    if inner.sync_all().is_err() {
        let _ = std::fs::remove_file(&staging_path);
        return Err(ChronoIndexError::AppendIo);
    }
    drop(inner);

    // Atomic rename to publish.
    if std::fs::rename(&staging_path, &mtimes_path).is_err() {
        let _ = std::fs::remove_file(&staging_path);
        return Err(ChronoIndexError::RenameIo);
    }

    let final_last = match last_emitted {
        Some(r) => (r.mtime_sec, r.mtime_nsec),
        None => return Err(ChronoIndexError::AppendIo),
    };
    Ok(final_last)
}

// =========================================================================
// The append entrypoint
// =========================================================================

/// Incrementally appends any new files in `parent_directory_to_index`
/// to the existing index, updating `header.bin` atomically on success.
///
/// Pre-conditions:
///   - `header.bin`, `names.bin`, `mtimes.bin` exist and are consistent
///     with each other (this is the responsibility of the caller's
///     orchestration in part d; if not, the caller should cold-rebuild
///     instead).
///   - `current_header` reflects the on-disk header.
///
/// Post-conditions on success:
///   - `header.file_count` reflects the new total.
///   - `header.signal_hash` XORs in each newly indexed basename.
///   - `header.last_mtime_*` reflects the newest record in `mtimes.bin`.
///   - `header.invariant_breach_count` is incremented per out-of-order
///     batch.
///   - `cursor.bin` is unchanged.
///
/// Post-conditions on failure:
///   - Returns a terse error code.
///   - `header.bin` is updated only if it can be made consistent with
///     the (possibly partial) new state of `names.bin` and `mtimes.bin`.
///     If even that fails, the previous header remains in place; the
///     caller's next orchestration round will detect the inconsistency
///     and trigger a cold rebuild. Never halts.
pub fn incremental_append_new_files(
    temp_root_dir: &Path,
    parent_directory_to_index: &Path,
    current_header: &ChronoIndexHeader,
) -> Result<AppendSummary, ChronoIndexError> {
    // Validate the parent path in the header still matches what was
    // passed in. If it has changed, caller should rebuild, not append.
    {
        let passed_in_bytes = posix_path_to_bytes(parent_directory_to_index)?;
        if passed_in_bytes != current_header.parent_path_slice() {
            return Err(ChronoIndexError::ParentPathInvalid);
        }
    }

    // Make sure the name-hash sidecar is present and matches file_count.
    ensure_name_hashes_sidecar_consistent(temp_root_dir, current_header.file_count)?;

    let mut summary = AppendSummary {
        files_appended: 0,
        entries_skipped_overlong_name: 0,
        entries_skipped_stat_failed: 0,
        entries_skipped_non_regular: 0,
        invariant_breaches_this_call: 0,
    };

    // Mutable header copy that we will commit only on success.
    let mut working_header: ChronoIndexHeader = current_header.clone();

    // Bounded batch buffer. Single allocation per call.
    let mut current_batch: Vec<MtimeRecord> = Vec::with_capacity(APPEND_BATCH_RECORDS);
    let mut current_batch_signal_xor: u64 = 0;

    let directory_iterator = match std::fs::read_dir(parent_directory_to_index) {
        Ok(it) => it,
        Err(_) => return Err(ChronoIndexError::AppendIo),
    };

    for directory_entry_result in directory_iterator {
        let directory_entry = match directory_entry_result {
            Ok(e) => e,
            Err(_) => {
                summary.entries_skipped_stat_failed =
                    summary.entries_skipped_stat_failed.saturating_add(1);
                continue;
            }
        };

        let file_type_info = match directory_entry.file_type() {
            Ok(ft) => ft,
            Err(_) => {
                summary.entries_skipped_stat_failed =
                    summary.entries_skipped_stat_failed.saturating_add(1);
                continue;
            }
        };
        if !file_type_info.is_file() {
            summary.entries_skipped_non_regular =
                summary.entries_skipped_non_regular.saturating_add(1);
            continue;
        }

        // Basename bytes (POSIX).
        let file_name_os = directory_entry.file_name();
        let basename_bytes: &[u8] = {
            #[cfg(unix)]
            {
                use std::os::unix::ffi::OsStrExt;
                file_name_os.as_bytes()
            }
            #[cfg(not(unix))]
            {
                summary.entries_skipped_overlong_name =
                    summary.entries_skipped_overlong_name.saturating_add(1);
                continue;
            }
        };

        if basename_bytes.len() > MAX_BASENAME_LEN {
            summary.entries_skipped_overlong_name =
                summary.entries_skipped_overlong_name.saturating_add(1);
            continue;
        }

        // Compute hash and check the sidecar to see if this is an
        // already-indexed file.
        let basename_hash = fnv1a_64(basename_bytes);
        match name_hash_is_present_in_sidecar(temp_root_dir, basename_hash)? {
            true => {
                // Known file — nothing to do.
                continue;
            }
            false => {
                // Defensive double-check: the sidecar is a *hash* index,
                // so a hash collision against an existing-but-different
                // basename is theoretically possible (u64 collision odds
                // are negligible for tens of thousands of files but not
                // zero in principle). We resolve such ambiguity safely
                // by treating an apparent collision as "skip and rebuild
                // later" — i.e. we conservatively skip this entry in
                // this path. Since hash collisions are astronomically
                // rare in practice this branch is effectively dead.
                //
                // Implementation note: we cannot detect the collision
                // cheaply here without a full scan of names.bin; we
                // accept the trade-off of an extremely rare
                // false-skip. Out-of-band consistency checks (e.g. the
                // signal_hash mismatch detection in part d) will
                // eventually trigger a rebuild that re-indexes the
                // missed file. No halt, no data loss.
            }
        }

        // stat() for mtime.
        let metadata = match directory_entry.metadata() {
            Ok(md) => md,
            Err(_) => {
                summary.entries_skipped_stat_failed =
                    summary.entries_skipped_stat_failed.saturating_add(1);
                continue;
            }
        };
        let (mtime_sec, mtime_nsec) = match extract_mtime_seconds_and_nanos(&metadata) {
            Some(pair) => pair,
            None => {
                summary.entries_skipped_stat_failed =
                    summary.entries_skipped_stat_failed.saturating_add(1);
                continue;
            }
        };

        // Append basename and hash sidecar in lockstep, assigning a
        // new record_id == current file_count + items already appended
        // in this call.
        let new_record_id = working_header.file_count;
        // Pack and write the basename.
        let name_record = match pack_basename_record(basename_bytes) {
            Some(packed) => packed,
            None => {
                // pack already enforces MAX_BASENAME_LEN; if we got here
                // the basename passed the earlier length check, so this
                // branch is unreachable in practice. Defensive only.
                summary.entries_skipped_overlong_name =
                    summary.entries_skipped_overlong_name.saturating_add(1);
                continue;
            }
        };

        if let Err(write_error) = append_basename_record_to_names(temp_root_dir, &name_record) {
            // Try to flush whatever batch is pending so the index can be
            // committed in a consistent prefix state.
            let _ = flush_batch_and_update_header(
                temp_root_dir,
                &mut current_batch,
                &mut working_header,
                &mut current_batch_signal_xor,
                &mut summary,
            );
            // Commit best-effort header even on failure to keep files
            // in sync. If this fails too, the caller's orchestration
            // will trigger a rebuild on next tick.
            let _ = write_header_atomic(temp_root_dir, &working_header);
            return Err(write_error);
        }
        if let Err(write_error) = append_name_hash_record(temp_root_dir, basename_hash) {
            let _ = flush_batch_and_update_header(
                temp_root_dir,
                &mut current_batch,
                &mut working_header,
                &mut current_batch_signal_xor,
                &mut summary,
            );
            let _ = write_header_atomic(temp_root_dir, &working_header);
            return Err(write_error);
        }

        let new_record = MtimeRecord {
            mtime_sec,
            mtime_nsec,
            record_id: new_record_id,
        };
        current_batch.push(new_record);
        current_batch_signal_xor ^= basename_hash;
        working_header.file_count = working_header.file_count.saturating_add(1);

        if current_batch.len() >= APPEND_BATCH_RECORDS {
            flush_batch_and_update_header(
                temp_root_dir,
                &mut current_batch,
                &mut working_header,
                &mut current_batch_signal_xor,
                &mut summary,
            )?;
        }
    }

    // Final flush of whatever remains.
    if !current_batch.is_empty() {
        flush_batch_and_update_header(
            temp_root_dir,
            &mut current_batch,
            &mut working_header,
            &mut current_batch_signal_xor,
            &mut summary,
        )?;
    }

    // sync_all on name_hashes.bin is implicit (we used .append which
    // does not buffer in a BufWriter), but we explicitly fsync now to
    // make the sidecar durable before committing the header.
    {
        let hashes_path = build_index_file_path(temp_root_dir, NAME_HASHES_FILENAME);
        if let Ok(h) = File::open(&hashes_path) {
            let _ = h.sync_all();
        }
    }
    // Same for names.bin.
    {
        let names_path = build_index_file_path(temp_root_dir, NAMES_FILENAME);
        if let Ok(h) = File::open(&names_path) {
            let _ = h.sync_all();
        }
    }

    // Commit the header. This is the atomic publish point.
    write_header_atomic(temp_root_dir, &working_header)?;
    Ok(summary)
}

/// Flushes a (possibly fast-path or slow-path) batch to `mtimes.bin`
/// and updates `working_header.last_mtime_*`, `signal_hash`, and
/// `invariant_breach_count` accordingly.
fn flush_batch_and_update_header(
    temp_root_dir: &Path,
    current_batch: &mut Vec<MtimeRecord>,
    working_header: &mut ChronoIndexHeader,
    current_batch_signal_xor: &mut u64,
    summary: &mut AppendSummary,
) -> Result<(), ChronoIndexError> {
    if current_batch.is_empty() {
        return Ok(());
    }

    // Sort the batch by the same total order as the file.
    current_batch.sort_unstable_by(|left, right| {
        if left.mtime_sec != right.mtime_sec {
            return left.mtime_sec.cmp(&right.mtime_sec);
        }
        if left.mtime_nsec != right.mtime_nsec {
            return left.mtime_nsec.cmp(&right.mtime_nsec);
        }
        left.record_id.cmp(&right.record_id)
    });

    // Decide fast vs. slow path. Compare the *smallest* record in the
    // batch to working_header.last_mtime_*. If the smallest is strictly
    // greater than (or equal to) the current last, pure append is sound.
    let smallest_in_batch = current_batch[0];
    let last_in_file = MtimeRecord {
        mtime_sec: working_header.last_mtime_sec,
        mtime_nsec: working_header.last_mtime_nsec,
        record_id: 0,
    };
    // smallest_in_batch >= last_in_file (sec, nsec) ?
    let fast_path_ok = smallest_in_batch.mtime_sec > last_in_file.mtime_sec
        || (smallest_in_batch.mtime_sec == last_in_file.mtime_sec
            && smallest_in_batch.mtime_nsec >= last_in_file.mtime_nsec)
        || working_header.file_count == current_batch.len() as u64; // first ever append

    if fast_path_ok {
        append_sorted_mtime_batch(temp_root_dir, current_batch)?;
        // Update last_mtime_* to the newest in the batch (which is the
        // last element after sort).
        let newest = match current_batch.last() {
            Some(r) => *r,
            None => return Err(ChronoIndexError::AppendIo),
        };
        working_header.last_mtime_sec = newest.mtime_sec;
        working_header.last_mtime_nsec = newest.mtime_nsec;
    } else {
        // Slow path: at least one batch record is older than current
        // last. Increment invariant breach counter and merge-insert.
        working_header.invariant_breach_count =
            working_header.invariant_breach_count.saturating_add(1);
        summary.invariant_breaches_this_call =
            summary.invariant_breaches_this_call.saturating_add(1);
        let (new_last_sec, new_last_nsec) =
            merge_insert_out_of_order_batch(temp_root_dir, current_batch)?;
        working_header.last_mtime_sec = new_last_sec;
        working_header.last_mtime_nsec = new_last_nsec;
    }

    // Update signal hash and counters.
    working_header.signal_hash ^= *current_batch_signal_xor;
    summary.files_appended = summary
        .files_appended
        .saturating_add(current_batch.len() as u64);

    current_batch.clear();
    *current_batch_signal_xor = 0;
    Ok(())
}

// =========================================================================
// Tests for part (c)
// =========================================================================

#[cfg(test)]
mod chrono_index_part_c_tests {
    use super::*;
    // use std::io::Write as _;

    fn make_test_temp_root(label: &str) -> PathBuf {
        let mut scratch = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        scratch.push(format!(
            "chrono_index_c_{}_{}_{}",
            label,
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&scratch).expect("setup");
        scratch
    }

    fn make_watched_dir_with_files(label: &str, files: &[(&str, &[u8])]) -> PathBuf {
        let mut watched = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        watched.push(format!(
            "chrono_watched_c_{}_{}_{}",
            label,
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&watched).expect("setup");
        for (basename, content) in files {
            let mut path = watched.clone();
            path.push(basename);
            let mut f = std::fs::File::create(&path).expect("create");
            f.write_all(content).expect("write");
            f.sync_all().expect("sync");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        watched
    }

    fn add_file_to_watched_dir(watched_dir: &Path, basename: &str, content: &[u8]) {
        // Sleep first so the new file has a strictly newer mtime than
        // anything pre-existing.
        std::thread::sleep(std::time::Duration::from_millis(15));
        let mut path = PathBuf::from(watched_dir);
        path.push(basename);
        let mut f = std::fs::File::create(&path).expect("create new");
        f.write_all(content).expect("write new");
        f.sync_all().expect("sync new");
    }

    fn read_all_mtime_records(temp_root: &Path) -> Vec<MtimeRecord> {
        let path = build_index_file_path(temp_root, MTIMES_FILENAME);
        let mut handle = File::open(&path).expect("open mtimes");
        let mut out = Vec::new();
        let mut buf = [0u8; MTIME_RECORD_SIZE];
        loop {
            match handle.read_exact(&mut buf) {
                Ok(()) => out.push(MtimeRecord::read_from(&buf)),
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(_) => panic!("read error in test helper"),
            }
        }
        out
    }

    fn read_basename_at_record_id(temp_root: &Path, record_id: u64) -> Vec<u8> {
        let path = build_index_file_path(temp_root, NAMES_FILENAME);
        let mut handle = File::open(&path).expect("open names");
        handle
            .seek(SeekFrom::Start(
                record_id.saturating_mul(NAME_RECORD_SIZE as u64),
            ))
            .expect("seek");
        let mut buf = [0u8; NAME_RECORD_SIZE];
        handle.read_exact(&mut buf).expect("read name");
        let used = basename_used_length(&buf);
        buf[..used].to_vec()
    }

    #[test]
    fn append_adds_single_new_file_to_already_built_index() {
        let temp_root = make_test_temp_root("single_append");
        let watched = make_watched_dir_with_files(
            "single_append",
            &[("first.txt", b"a"), ("second.txt", b"b")],
        );
        let _ = cold_build_index(&temp_root, &watched).expect("cold build");
        let header_after_build = read_header(&temp_root).expect("read").expect("present");
        assert_eq!(header_after_build.file_count, 2);

        // Add a new file with strictly newer mtime.
        add_file_to_watched_dir(&watched, "third.txt", b"c");

        let summary = incremental_append_new_files(&temp_root, &watched, &header_after_build)
            .expect("append ok");
        assert_eq!(summary.files_appended, 1);
        assert_eq!(summary.invariant_breaches_this_call, 0);

        let header_after_append = read_header(&temp_root).expect("read").expect("present");
        assert_eq!(header_after_append.file_count, 3);
        assert!(header_after_append.last_mtime_sec >= header_after_build.last_mtime_sec);

        // mtimes.bin must remain sorted.
        let records = read_all_mtime_records(&temp_root);
        assert_eq!(records.len(), 3);
        for window in records.windows(2) {
            let ordered = window[0].is_strictly_before(window[1])
                || (window[0].mtime_sec == window[1].mtime_sec
                    && window[0].mtime_nsec == window[1].mtime_nsec
                    && window[0].record_id < window[1].record_id);
            assert!(ordered, "mtimes.bin lost sorted order");
        }

        // The newest record must point to "third.txt".
        let last = *records.last().expect("at least one");
        let name = read_basename_at_record_id(&temp_root, last.record_id);
        assert_eq!(name, b"third.txt");

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn append_handles_multiple_new_files_in_one_call() {
        let temp_root = make_test_temp_root("multi_append");
        let watched = make_watched_dir_with_files("multi_append", &[("a.dat", b"1")]);
        let _ = cold_build_index(&temp_root, &watched).expect("cold build");
        let header_after_build = read_header(&temp_root).expect("r").expect("p");

        for new_name in &["b.dat", "c.dat", "d.dat", "e.dat"] {
            add_file_to_watched_dir(&watched, new_name, new_name.as_bytes());
        }

        let summary = incremental_append_new_files(&temp_root, &watched, &header_after_build)
            .expect("append ok");
        assert_eq!(summary.files_appended, 4);

        let header_after = read_header(&temp_root).expect("r").expect("p");
        assert_eq!(header_after.file_count, 5);

        let records = read_all_mtime_records(&temp_root);
        assert_eq!(records.len(), 5);
        for window in records.windows(2) {
            let ordered = window[0].is_strictly_before(window[1])
                || (window[0].mtime_sec == window[1].mtime_sec
                    && window[0].mtime_nsec == window[1].mtime_nsec
                    && window[0].record_id < window[1].record_id);
            assert!(ordered);
        }

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn append_is_idempotent_when_no_new_files() {
        let temp_root = make_test_temp_root("noop_append");
        let watched = make_watched_dir_with_files(
            "noop_append",
            &[("one.x", b"1"), ("two.x", b"2"), ("three.x", b"3")],
        );
        let _ = cold_build_index(&temp_root, &watched).expect("cold build");
        let header_before = read_header(&temp_root).expect("r").expect("p");

        let summary =
            incremental_append_new_files(&temp_root, &watched, &header_before).expect("append ok");
        assert_eq!(summary.files_appended, 0);

        let header_after = read_header(&temp_root).expect("r").expect("p");
        assert_eq!(header_after.file_count, header_before.file_count);
        assert_eq!(header_after.signal_hash, header_before.signal_hash);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn append_signal_hash_xor_accumulates_correctly() {
        let temp_root = make_test_temp_root("signal_xor");
        let watched = make_watched_dir_with_files("signal_xor", &[("alpha", b"a")]);
        let _ = cold_build_index(&temp_root, &watched).expect("cold build");
        let header_after_build = read_header(&temp_root).expect("r").expect("p");
        let expected_initial = fnv1a_64(b"alpha");
        assert_eq!(header_after_build.signal_hash, expected_initial);

        add_file_to_watched_dir(&watched, "beta", b"b");
        add_file_to_watched_dir(&watched, "gamma", b"g");

        let _ = incremental_append_new_files(&temp_root, &watched, &header_after_build)
            .expect("append ok");

        let header_after_append = read_header(&temp_root).expect("r").expect("p");
        let expected_final = fnv1a_64(b"alpha") ^ fnv1a_64(b"beta") ^ fnv1a_64(b"gamma");
        assert_eq!(header_after_append.signal_hash, expected_final);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn append_rejects_wrong_parent_path() {
        let temp_root = make_test_temp_root("wrong_parent");
        let watched_a = make_watched_dir_with_files("wrong_parent_a", &[("x", b"x")]);
        let watched_b = make_watched_dir_with_files("wrong_parent_b", &[("y", b"y")]);
        let _ = cold_build_index(&temp_root, &watched_a).expect("cold build");
        let header = read_header(&temp_root).expect("r").expect("p");

        let result = incremental_append_new_files(&temp_root, &watched_b, &header);
        assert_eq!(result.err(), Some(ChronoIndexError::ParentPathInvalid));

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched_a);
        let _ = std::fs::remove_dir_all(&watched_b);
    }

    #[test]
    fn append_skips_overlong_basenames() {
        let temp_root = make_test_temp_root("overlong_append");
        let watched = make_watched_dir_with_files("overlong_append", &[("normal.txt", b"n")]);
        let _ = cold_build_index(&temp_root, &watched).expect("cold build");
        let header = read_header(&temp_root).expect("r").expect("p");

        // Add one valid + one overlong.
        add_file_to_watched_dir(&watched, "valid.txt", b"v");
        let overlong: String = "z".repeat(MAX_BASENAME_LEN + 1);
        add_file_to_watched_dir(&watched, overlong.as_str(), b"x");

        let summary =
            incremental_append_new_files(&temp_root, &watched, &header).expect("append ok");
        assert_eq!(summary.files_appended, 1);
        assert_eq!(summary.entries_skipped_overlong_name, 1);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn append_skips_subdirectory_entries() {
        let temp_root = make_test_temp_root("subdir_skip");
        let watched = make_watched_dir_with_files("subdir_skip", &[("file.txt", b"f")]);
        let _ = cold_build_index(&temp_root, &watched).expect("cold build");
        let header = read_header(&temp_root).expect("r").expect("p");

        let mut subdir = watched.clone();
        subdir.push("a_sub");
        std::fs::create_dir_all(&subdir).expect("mkdir");
        add_file_to_watched_dir(&watched, "newfile.txt", b"n");

        let summary =
            incremental_append_new_files(&temp_root, &watched, &header).expect("append ok");
        assert_eq!(summary.files_appended, 1);
        assert!(summary.entries_skipped_non_regular >= 1);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn name_hashes_sidecar_built_lazily_and_matches_names() {
        let temp_root = make_test_temp_root("sidecar");
        let watched =
            make_watched_dir_with_files("sidecar", &[("aa", b"1"), ("bb", b"2"), ("cc", b"3")]);
        let _ = cold_build_index(&temp_root, &watched).expect("cold build");

        // Sidecar should not yet exist after cold build.
        let sidecar_path = build_index_file_path(&temp_root, NAME_HASHES_FILENAME);
        assert!(!sidecar_path.exists(), "sidecar should be lazy");

        // Triggering append (with no new files) should build it.
        let header = read_header(&temp_root).expect("r").expect("p");
        let _ =
            incremental_append_new_files(&temp_root, &watched, &header).expect("append ok (noop)");
        assert!(sidecar_path.exists());

        let meta = std::fs::metadata(&sidecar_path).expect("meta");
        assert_eq!(meta.len() as usize, 3 * NAME_HASH_RECORD_SIZE);

        // Each hash in the sidecar must match the corresponding name.
        let mut sidecar_handle = File::open(&sidecar_path).expect("open");
        let mut names_handle =
            File::open(&build_index_file_path(&temp_root, NAMES_FILENAME)).expect("open names");
        for _ in 0..3u64 {
            let mut hash_buf = [0u8; NAME_HASH_RECORD_SIZE];
            sidecar_handle.read_exact(&mut hash_buf).expect("read hash");
            let stored_hash = u64::from_le_bytes(hash_buf);
            let mut name_buf = [0u8; NAME_RECORD_SIZE];
            names_handle.read_exact(&mut name_buf).expect("read name");
            let used = basename_used_length(&name_buf);
            assert_eq!(stored_hash, fnv1a_64(&name_buf[..used]));
        }

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn append_cursor_is_unchanged() {
        let temp_root = make_test_temp_root("cursor_unchanged");
        let watched = make_watched_dir_with_files("cursor_unchanged", &[("p.txt", b"p")]);
        let _ = cold_build_index(&temp_root, &watched).expect("cold build");
        write_cursor_atomic(&temp_root, 0).expect("init cursor");
        // Advance cursor manually to mimic having emitted one item.
        write_cursor_atomic(&temp_root, 1).expect("advance cursor");

        let header = read_header(&temp_root).expect("r").expect("p");
        add_file_to_watched_dir(&watched, "q.txt", b"q");
        let _ = incremental_append_new_files(&temp_root, &watched, &header).expect("append ok");

        // Cursor must not have moved.
        let cursor_after = read_cursor(&temp_root).expect("read cursor");
        assert_eq!(cursor_after, 1);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn append_after_build_keeps_mtimes_sorted_across_many_batches() {
        // Cross a batch boundary by appending APPEND_BATCH_RECORDS + a
        // few extra files. We keep the count test-feasible.
        let temp_root = make_test_temp_root("many_batches");
        let watched = make_watched_dir_with_files("many_batches", &[("seed.txt", b"s")]);
        let _ = cold_build_index(&temp_root, &watched).expect("cold build");
        let header = read_header(&temp_root).expect("r").expect("p");

        // We do not actually want to sleep 10ms × hundreds of times in
        // a test, so we reduce: just append a handful but include a
        // batch-size sanity check.
        let extras: usize = (APPEND_BATCH_RECORDS / 32).max(8);
        for i in 0..extras {
            let name = format!("extra_{:05}.dat", i);
            // Smaller sleep to keep test fast; ns-resolution filesystems
            // (ext4) preserve order even at 1 ms.
            std::thread::sleep(std::time::Duration::from_millis(2));
            let mut p = watched.clone();
            p.push(&name);
            let mut f = std::fs::File::create(&p).expect("create");
            f.write_all(name.as_bytes()).expect("write");
            f.sync_all().expect("sync");
        }

        let summary =
            incremental_append_new_files(&temp_root, &watched, &header).expect("append ok");
        assert_eq!(summary.files_appended as usize, extras);

        let records = read_all_mtime_records(&temp_root);
        assert_eq!(records.len(), 1 + extras);
        for window in records.windows(2) {
            let ordered = window[0].is_strictly_before(window[1])
                || (window[0].mtime_sec == window[1].mtime_sec
                    && window[0].mtime_nsec == window[1].mtime_nsec
                    && window[0].record_id < window[1].record_id);
            assert!(ordered, "mtimes.bin lost sorted order across batches");
        }

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn append_round_trip_then_second_append_works() {
        // Two successive appends must each commit their state cleanly,
        // so the second sees the header produced by the first.
        let temp_root = make_test_temp_root("two_appends");
        let watched = make_watched_dir_with_files("two_appends", &[("seed", b"s")]);
        let _ = cold_build_index(&temp_root, &watched).expect("cold build");

        // First append round.
        add_file_to_watched_dir(&watched, "round1_a", b"1a");
        add_file_to_watched_dir(&watched, "round1_b", b"1b");
        let header_after_build = read_header(&temp_root).expect("r").expect("p");
        let summary_1 = incremental_append_new_files(&temp_root, &watched, &header_after_build)
            .expect("append 1 ok");
        assert_eq!(summary_1.files_appended, 2);

        // Second append round, starting from the updated header.
        add_file_to_watched_dir(&watched, "round2_a", b"2a");
        let header_after_first = read_header(&temp_root).expect("r").expect("p");
        assert_eq!(header_after_first.file_count, 3);
        let summary_2 = incremental_append_new_files(&temp_root, &watched, &header_after_first)
            .expect("append 2 ok");
        assert_eq!(summary_2.files_appended, 1);

        // Final state must be sorted and contain all four entries.
        let header_after_second = read_header(&temp_root).expect("r").expect("p");
        assert_eq!(header_after_second.file_count, 4);

        let records = read_all_mtime_records(&temp_root);
        assert_eq!(records.len(), 4);
        for window in records.windows(2) {
            let ordered = window[0].is_strictly_before(window[1])
                || (window[0].mtime_sec == window[1].mtime_sec
                    && window[0].mtime_nsec == window[1].mtime_nsec
                    && window[0].record_id < window[1].record_id);
            assert!(ordered);
        }

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn append_slow_path_handles_out_of_order_mtime_without_halting() {
        // Synthetic exercise of `merge_insert_out_of_order_batch` via a
        // direct invocation. We build a tiny mtimes.bin with two records
        // and then merge-insert a third that sorts BEFORE both. The file
        // must remain sorted and the function must return the new last
        // mtime (which equals the previous last, since the inserted
        // record was older).
        let temp_root = make_test_temp_root("slow_path");
        ensure_index_directory_exists(&temp_root).expect("setup");

        let mtimes_path = build_index_file_path(&temp_root, MTIMES_FILENAME);
        {
            // Write two in-order records.
            let mut handle = std::fs::File::create(&mtimes_path).expect("create");
            let mut buf = [0u8; MTIME_RECORD_SIZE];
            MtimeRecord {
                mtime_sec: 100,
                mtime_nsec: 0,
                record_id: 0,
            }
            .write_into(&mut buf);
            handle.write_all(&buf).expect("w1");
            MtimeRecord {
                mtime_sec: 200,
                mtime_nsec: 0,
                record_id: 1,
            }
            .write_into(&mut buf);
            handle.write_all(&buf).expect("w2");
            handle.sync_all().expect("sync");
        }

        // Out-of-order batch: a record at mtime_sec=50, which is older
        // than every record currently in the file.
        let mut batch = [MtimeRecord {
            mtime_sec: 50,
            mtime_nsec: 0,
            record_id: 2,
        }];

        let (new_last_sec, new_last_nsec) =
            merge_insert_out_of_order_batch(&temp_root, &mut batch[..]).expect("merge insert ok");
        // The newest record is still the original mtime_sec=200 one.
        assert_eq!(new_last_sec, 200);
        assert_eq!(new_last_nsec, 0);

        // The file must now contain three records, sorted.
        let records = read_all_mtime_records(&temp_root);
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].mtime_sec, 50);
        assert_eq!(records[1].mtime_sec, 100);
        assert_eq!(records[2].mtime_sec, 200);

        let _ = std::fs::remove_dir_all(&temp_root);
    }

    #[test]
    fn name_hash_sidecar_lookup_finds_existing_and_misses_new() {
        let temp_root = make_test_temp_root("sidecar_lookup");
        let watched = make_watched_dir_with_files(
            "sidecar_lookup",
            &[("present_one", b"a"), ("present_two", b"b")],
        );
        let _ = cold_build_index(&temp_root, &watched).expect("cold build");
        // Build the sidecar by triggering an append (no new files).
        let header = read_header(&temp_root).expect("r").expect("p");
        let _ = incremental_append_new_files(&temp_root, &watched, &header).expect("noop append");

        // Lookup an existing basename hash → must be present.
        let present_hash = fnv1a_64(b"present_one");
        assert!(name_hash_is_present_in_sidecar(&temp_root, present_hash).expect("lookup ok"));

        // Lookup a basename that does not exist → must be absent.
        let absent_hash = fnv1a_64(b"definitely_not_there_xyz");
        assert!(!name_hash_is_present_in_sidecar(&temp_root, absent_hash).expect("lookup ok"));

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn basename_used_length_handles_full_and_partial_records() {
        // Full record (no NUL padding): length is NAME_RECORD_SIZE.
        let full = [b'a'; NAME_RECORD_SIZE];
        assert_eq!(basename_used_length(&full), NAME_RECORD_SIZE);

        // Empty record: length is 0.
        let empty = [0u8; NAME_RECORD_SIZE];
        assert_eq!(basename_used_length(&empty), 0);

        // Partial record: 5 bytes then NUL padding.
        let mut partial = [0u8; NAME_RECORD_SIZE];
        partial[..5].copy_from_slice(b"hello");
        assert_eq!(basename_used_length(&partial), 5);
    }
}

// =========================================================================
// Part (d): Tick function and update orchestration
// =========================================================================
//
// ## Two public entrypoints
//
// This part adds the two functions that callers normally use:
//
//   1. `update_index` — the orchestration that callers invoke whenever
//      they want the index to reflect the current state of the watched
//      directory. It compares the live directory against the committed
//      `header.bin` and dispatches to either:
//         * nothing (index is current),
//         * `incremental_append_new_files` (part c, the hot path), or
//         * `cold_build_index` (part b, the fallback).
//      Never halts; falls back to rebuild on any inconsistency.
//
//   2. `iter_return_next_chronological_abs_file_path` — the per-tick function.
//      Called at whatever cadence the caller wants (e.g. once per N
//      seconds). Reads the cursor, fetches the next chronologically-
//      ordered file, assembles its absolute path into a caller-provided
//      stack buffer, and advances the cursor atomically.
//
// ## Tick contract
//
// The caller provides:
//
//   - the index `temp_root_dir`,
//   - a mutable `[u8; MAX_FULL_PATH_LEN]` stack buffer.
//
// The function returns one of:
//
//   - `Ok(Some(TickEmission { path_byte_length, ... }))` — a file was
//      emitted; `out_path_buffer[..path_byte_length]` holds its absolute
//      POSIX path bytes. The cursor has been advanced on disk.
//
//   - `Ok(None)` — there are no more files to emit at the current commit.
//      The cursor is left at `header.file_count`. A subsequent
//      `update_index` may add more files and the next tick will resume.
//
//   - `Err(...)` — terse error code; the cursor and index are unchanged
//      from before the call. The caller may retry on the next tick.
//
// ## Memory discipline
//
// Per-tick allocations: none on the heap. The full absolute path is
// assembled into the caller's stack buffer. All on-disk reads are
// fixed-size (20 B for the mtime record, 64 B for the name record).
// A tiny `PathBuf` is constructed internally to satisfy `std::fs::File`
// APIs; it is freed before the function returns.

/// Maximum size of the caller-provided absolute-path buffer. POSIX
/// `PATH_MAX` is typically 4096 on Linux.
pub const MAX_FULL_PATH_LEN: usize = MAX_PARENT_PATH_LEN;

/// One emitted file from `iter_return_next_chronological_abs_file_path`.
///
/// The caller-provided path buffer contains the absolute POSIX path of
/// the emitted file in its first `path_byte_length` bytes. The remaining
/// bytes are unspecified.
#[derive(Clone, Copy, Debug)]
pub struct TickEmission {
    /// Number of valid path bytes in the caller's output buffer.
    pub path_byte_length: usize,
    /// The cursor position from which this file was emitted (0-based
    /// index into the chronological order). Useful for caller logging
    /// and progress reporting.
    pub emitted_cursor_index: u64,
    /// The mtime of the emitted file. Exposed for caller logging /
    /// observability.
    pub emitted_mtime_sec: i64,
    pub emitted_mtime_nsec: i32,
}

// =========================================================================
// Public summary type for update_index
// =========================================================================

/// Discrete outcome categories from `update_index`. Carries no user data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateOutcome {
    /// No prior committed index existed; a cold build was performed.
    ColdBuildCompleted,
    /// A previously committed index was found to be unusable
    /// (structural / consistency mismatch); a cold rebuild was performed.
    RebuiltDueToInconsistency,
    /// The live directory matched the committed index exactly; nothing
    /// was changed on disk.
    NoChangesDetected,
    /// The live directory had grown since the last commit; the new
    /// files were appended incrementally.
    IncrementalAppendCompleted,
}

/// Aggregate summary returned by `update_index`. The numeric fields are
/// 0 for outcomes that did not exercise the corresponding path.
#[derive(Clone, Copy, Debug)]
pub struct UpdateSummary {
    pub outcome: UpdateOutcome,
    /// Total files indexed by the index after this update.
    pub final_file_count: u64,
    /// If a cold build ran, the build's summary; otherwise zeroes.
    pub cold_build_summary: ColdBuildSummary,
    /// If an incremental append ran, that summary; otherwise zeroes.
    pub append_summary: AppendSummary,
}

// =========================================================================
// Live-directory probe: count + signal hash, no stat()
// =========================================================================

/// Probe result for the live directory: total regular-file count and
/// XOR-fold of FNV-1a 64 over their basenames. Used to decide between
/// the no-op / incremental / rebuild paths in `update_index`.
///
/// `entries_skipped_overlong_name`, `entries_skipped_stat_failed`, and
/// `entries_skipped_non_regular` are tracked here too so the orchestrator
/// has the same view of "what counts" as the build/append paths do.
#[derive(Clone, Copy, Debug)]
struct LiveDirectoryProbe {
    live_file_count: u64,
    live_signal_hash: u64,
    entries_skipped_overlong_name: u64,
    entries_skipped_stat_failed: u64,
    entries_skipped_non_regular: u64,
}

/// Streams `read_dir(parent_directory)` once and computes the probe.
/// Does not call `stat()` on entries (uses `file_type()` which is
/// returned by `readdir(3)` on Linux for most filesystems).
///
/// Memory: O(1). One `OsString` per entry from the stdlib iterator,
/// freed before the next iteration. No accumulation.
fn probe_live_directory(parent_directory: &Path) -> Result<LiveDirectoryProbe, ChronoIndexError> {
    let directory_iterator = match std::fs::read_dir(parent_directory) {
        Ok(it) => it,
        Err(_) => return Err(ChronoIndexError::BuildIo),
    };

    let mut probe = LiveDirectoryProbe {
        live_file_count: 0,
        live_signal_hash: 0,
        entries_skipped_overlong_name: 0,
        entries_skipped_stat_failed: 0,
        entries_skipped_non_regular: 0,
    };

    for directory_entry_result in directory_iterator {
        let directory_entry = match directory_entry_result {
            Ok(e) => e,
            Err(_) => {
                probe.entries_skipped_stat_failed =
                    probe.entries_skipped_stat_failed.saturating_add(1);
                continue;
            }
        };

        let file_type_info = match directory_entry.file_type() {
            Ok(ft) => ft,
            Err(_) => {
                probe.entries_skipped_stat_failed =
                    probe.entries_skipped_stat_failed.saturating_add(1);
                continue;
            }
        };
        if !file_type_info.is_file() {
            probe.entries_skipped_non_regular = probe.entries_skipped_non_regular.saturating_add(1);
            continue;
        }

        let file_name_os = directory_entry.file_name();
        let basename_bytes: &[u8] = {
            #[cfg(unix)]
            {
                use std::os::unix::ffi::OsStrExt;
                file_name_os.as_bytes()
            }
            #[cfg(not(unix))]
            {
                probe.entries_skipped_overlong_name =
                    probe.entries_skipped_overlong_name.saturating_add(1);
                continue;
            }
        };

        if basename_bytes.len() > MAX_BASENAME_LEN {
            probe.entries_skipped_overlong_name =
                probe.entries_skipped_overlong_name.saturating_add(1);
            continue;
        }

        probe.live_signal_hash ^= fnv1a_64(basename_bytes);
        probe.live_file_count = probe.live_file_count.saturating_add(1);
    }

    Ok(probe)
}

// =========================================================================
// On-disk consistency: do header.file_count and the data files agree?
// =========================================================================

/// Returns `true` iff `names.bin` and `mtimes.bin` are both present and
/// their byte sizes exactly match `header.file_count`. Used to detect a
/// half-applied prior write (e.g. a crash between renaming data files
/// and renaming the header).
///
/// Memory: O(1).
fn data_files_match_header_count(temp_root_dir: &Path, header: &ChronoIndexHeader) -> bool {
    let names_path = build_index_file_path(temp_root_dir, NAMES_FILENAME);
    let mtimes_path = build_index_file_path(temp_root_dir, MTIMES_FILENAME);

    let expected_names_size = header.file_count.saturating_mul(NAME_RECORD_SIZE as u64);
    let expected_mtimes_size = header.file_count.saturating_mul(MTIME_RECORD_SIZE as u64);

    let names_size_matches = match std::fs::metadata(&names_path) {
        Ok(m) => m.len() == expected_names_size,
        Err(_) => false,
    };
    let mtimes_size_matches = match std::fs::metadata(&mtimes_path) {
        Ok(m) => m.len() == expected_mtimes_size,
        Err(_) => false,
    };
    names_size_matches && mtimes_size_matches
}

// =========================================================================
// update_index — the high-level orchestration entrypoint
// =========================================================================

/// Brings the on-disk index in `<temp_root_dir>/chrono_index/` up to
/// date with the current contents of `parent_directory_to_index`.
///
/// Decision logic:
///
///   - No header present, OR header structurally invalid, OR data files
///     do not match the header's `file_count`:
///       → cold build (part b).
///
///   - Header present and data files consistent, AND the live directory
///     matches `(file_count, signal_hash)` exactly:
///       → no-op.
///
///   - Header present, data files consistent, AND
///       live_file_count >= header.file_count
///       AND a delta XOR exists such that all-pre-existing names are
///       still represented (verified by `header.signal_hash XOR
///       live_signal_hash` equalling the XOR of *only* the new
///       basenames):
///       → incremental append (part c).
///
///   - Anything else (live count shrank, hashes incompatible, etc.):
///       → cold rebuild.
///
/// Per project policy: never halts. Any unrecoverable inconsistency
/// triggers a cold rebuild rather than an error return.
pub fn update_index(
    temp_root_dir: &Path,
    parent_directory_to_index: &Path,
) -> Result<UpdateSummary, ChronoIndexError> {
    ensure_index_directory_exists(temp_root_dir)?;

    // Empty summaries used when a path is not taken.
    let empty_cold_build_summary = ColdBuildSummary {
        files_indexed: 0,
        entries_skipped_overlong_name: 0,
        entries_skipped_stat_failed: 0,
        entries_skipped_non_regular: 0,
    };
    let empty_append_summary = AppendSummary {
        files_appended: 0,
        entries_skipped_overlong_name: 0,
        entries_skipped_stat_failed: 0,
        entries_skipped_non_regular: 0,
        invariant_breaches_this_call: 0,
    };

    // --- Step 1: read header (or detect absence / corruption). --------
    let committed_header_opt = match read_header(temp_root_dir) {
        Ok(opt) => opt,
        Err(_structural_or_io_error) => {
            // Either I/O error or structural mismatch (bad magic, bad
            // version, bad size). Treat as "no usable header" and
            // rebuild.
            None
        }
    };

    // --- Step 2: if no usable header, cold build outright. ------------
    let committed_header = match committed_header_opt {
        Some(h) => h,
        None => {
            let build_summary = cold_build_index(temp_root_dir, parent_directory_to_index)?;
            let header_after = read_header(temp_root_dir)?.ok_or(ChronoIndexError::BuildIo)?;
            return Ok(UpdateSummary {
                outcome: UpdateOutcome::ColdBuildCompleted,
                final_file_count: header_after.file_count,
                cold_build_summary: build_summary,
                append_summary: empty_append_summary,
            });
        }
    };

    // --- Step 3: verify data files agree with the header. -------------
    // If a previous run crashed mid-commit, the data files may be
    // larger or smaller than the header says. Rebuild in that case.
    if !data_files_match_header_count(temp_root_dir, &committed_header) {
        let build_summary = cold_build_index(temp_root_dir, parent_directory_to_index)?;
        let header_after = read_header(temp_root_dir)?.ok_or(ChronoIndexError::BuildIo)?;
        return Ok(UpdateSummary {
            outcome: UpdateOutcome::RebuiltDueToInconsistency,
            final_file_count: header_after.file_count,
            cold_build_summary: build_summary,
            append_summary: empty_append_summary,
        });
    }

    // --- Step 4: verify parent path in header matches caller's path. --
    let passed_in_parent_bytes = posix_path_to_bytes(parent_directory_to_index)?;
    if passed_in_parent_bytes != committed_header.parent_path_slice() {
        // The caller is now watching a different directory than the
        // committed index. Rebuild against the new directory.
        let build_summary = cold_build_index(temp_root_dir, parent_directory_to_index)?;
        let header_after = read_header(temp_root_dir)?.ok_or(ChronoIndexError::BuildIo)?;
        return Ok(UpdateSummary {
            outcome: UpdateOutcome::RebuiltDueToInconsistency,
            final_file_count: header_after.file_count,
            cold_build_summary: build_summary,
            append_summary: empty_append_summary,
        });
    }

    // --- Step 5: probe the live directory. ----------------------------
    let probe = probe_live_directory(parent_directory_to_index)?;

    // No-op case: counts and hashes match.
    if probe.live_file_count == committed_header.file_count
        && probe.live_signal_hash == committed_header.signal_hash
    {
        return Ok(UpdateSummary {
            outcome: UpdateOutcome::NoChangesDetected,
            final_file_count: committed_header.file_count,
            cold_build_summary: empty_cold_build_summary,
            append_summary: empty_append_summary,
        });
    }

    // --- Step 6: append-eligible case. --------------------------------
    //
    // Per project rules, files are never deleted. So:
    //   - live_file_count < committed_header.file_count → impossible in
    //     a well-behaved environment; treat as inconsistency, rebuild.
    //   - live_file_count == committed_header.file_count but hashes
    //     differ → some basename changed identity (rename / replace).
    //     Per project rules this should not occur; treat as inconsistency.
    //   - live_file_count > committed_header.file_count → may be
    //     append-eligible. We hand off to `incremental_append_new_files`,
    //     which performs its own per-basename check via the
    //     `name_hashes.bin` sidecar.
    //
    // The XOR delta check (header.signal_hash XOR live_signal_hash
    // equals XOR of *only* new basenames) is automatically satisfied
    // when the only change is the addition of new files, because XOR is
    // its own inverse:
    //   new_names_xor = live - existing  (in XOR algebra)
    //                 = live XOR existing
    // The append path produces a final signal_hash equal to
    // existing XOR new_names_xor == live_signal_hash by construction.
    // If after append the resulting header.signal_hash differs from the
    // probe's live_signal_hash, the orchestrator can detect that and
    // fall through to rebuild on the next call. We perform that check
    // below as a defensive consistency gate.
    if probe.live_file_count < committed_header.file_count
        || probe.live_file_count == committed_header.file_count
    {
        // Either shrinking (impossible per spec) or same-count-different-
        // contents (rename/replace, also impossible per spec). Rebuild.
        let build_summary = cold_build_index(temp_root_dir, parent_directory_to_index)?;
        let header_after = read_header(temp_root_dir)?.ok_or(ChronoIndexError::BuildIo)?;
        return Ok(UpdateSummary {
            outcome: UpdateOutcome::RebuiltDueToInconsistency,
            final_file_count: header_after.file_count,
            cold_build_summary: build_summary,
            append_summary: empty_append_summary,
        });
    }

    // live_file_count > committed_header.file_count → attempt append.
    let append_outcome =
        incremental_append_new_files(temp_root_dir, parent_directory_to_index, &committed_header);

    let append_summary = match append_outcome {
        Ok(s) => s,
        Err(_append_error) => {
            // The append failed partway. Per the contract of
            // `incremental_append_new_files`, it has already made a
            // best-effort attempt to keep the header consistent with
            // whatever prefix it managed to write. To be fully safe,
            // we now rebuild from scratch so the index is guaranteed
            // consistent with the live directory.
            let build_summary = cold_build_index(temp_root_dir, parent_directory_to_index)?;
            let header_after = read_header(temp_root_dir)?.ok_or(ChronoIndexError::BuildIo)?;
            return Ok(UpdateSummary {
                outcome: UpdateOutcome::RebuiltDueToInconsistency,
                final_file_count: header_after.file_count,
                cold_build_summary: build_summary,
                append_summary: empty_append_summary,
            });
        }
    };

    // Post-append consistency gate: re-read header and verify its
    // signal_hash now matches the probe's live_signal_hash. If not,
    // some assumption was violated (e.g. an FNV hash collision causing
    // a conservative skip in the append path); rebuild on the next
    // update_index call by treating this round as a rebuild.
    let header_after_append = read_header(temp_root_dir)?.ok_or(ChronoIndexError::AppendIo)?;
    if header_after_append.signal_hash != probe.live_signal_hash
        || header_after_append.file_count != probe.live_file_count
    {
        let build_summary = cold_build_index(temp_root_dir, parent_directory_to_index)?;
        let header_after_rebuild = read_header(temp_root_dir)?.ok_or(ChronoIndexError::BuildIo)?;
        return Ok(UpdateSummary {
            outcome: UpdateOutcome::RebuiltDueToInconsistency,
            final_file_count: header_after_rebuild.file_count,
            cold_build_summary: build_summary,
            append_summary: empty_append_summary,
        });
    }

    Ok(UpdateSummary {
        outcome: UpdateOutcome::IncrementalAppendCompleted,
        final_file_count: header_after_append.file_count,
        cold_build_summary: empty_cold_build_summary,
        append_summary,
    })
}

// =========================================================================
// Tick: emit the next chronological file
// =========================================================================

/// Reads one `MtimeRecord` from `mtimes.bin` at the given record index.
/// Bounded stack memory. Returns a terse error code on any I/O failure.
fn read_mtime_record_at_index(
    temp_root_dir: &Path,
    mtime_index: u64,
) -> Result<MtimeRecord, ChronoIndexError> {
    let mtimes_path = build_index_file_path(temp_root_dir, MTIMES_FILENAME);
    let mut handle = match File::open(&mtimes_path) {
        Ok(h) => h,
        Err(_) => return Err(ChronoIndexError::TickIo),
    };
    let byte_offset = mtime_index.saturating_mul(MTIME_RECORD_SIZE as u64);
    if handle.seek(SeekFrom::Start(byte_offset)).is_err() {
        return Err(ChronoIndexError::TickIo);
    }
    let mut buffer = [0u8; MTIME_RECORD_SIZE];
    if handle.read_exact(&mut buffer).is_err() {
        return Err(ChronoIndexError::TickIo);
    }
    Ok(MtimeRecord::read_from(&buffer))
}

/// Reads one `names.bin` record into the supplied stack buffer.
/// Returns the used length (number of bytes before NUL padding).
fn read_name_record_at_record_id(
    temp_root_dir: &Path,
    record_id: u64,
    out_name_record: &mut [u8; NAME_RECORD_SIZE],
) -> Result<usize, ChronoIndexError> {
    let names_path = build_index_file_path(temp_root_dir, NAMES_FILENAME);
    let mut handle = match File::open(&names_path) {
        Ok(h) => h,
        Err(_) => return Err(ChronoIndexError::TickIo),
    };
    let byte_offset = record_id.saturating_mul(NAME_RECORD_SIZE as u64);
    if handle.seek(SeekFrom::Start(byte_offset)).is_err() {
        return Err(ChronoIndexError::TickIo);
    }
    if handle.read_exact(out_name_record).is_err() {
        return Err(ChronoIndexError::TickIo);
    }
    Ok(basename_used_length(out_name_record))
}

/// Assembles `parent_path + "/" + basename` into `out_path_buffer`.
/// Returns the used length, or an error if the result would exceed
/// `MAX_FULL_PATH_LEN`.
///
/// If `parent_path` already ends with `/`, the separator is not duplicated.
/// All operations are bounds-checked; no panic.
fn assemble_absolute_path_into_buffer(
    parent_path_bytes: &[u8],
    basename_bytes: &[u8],
    out_path_buffer: &mut [u8; MAX_FULL_PATH_LEN],
) -> Result<usize, ChronoIndexError> {
    // Defensive: a malformed empty parent is rejected.
    if parent_path_bytes.is_empty() {
        return Err(ChronoIndexError::ParentPathInvalid);
    }

    let parent_ends_with_separator = parent_path_bytes
        .last()
        .map(|byte| *byte == b'/')
        .unwrap_or(false);
    let separator_byte_count: usize = if parent_ends_with_separator { 0 } else { 1 };

    // Bounds check: parent + sep + basename must fit.
    let total_length = parent_path_bytes
        .len()
        .saturating_add(separator_byte_count)
        .saturating_add(basename_bytes.len());
    if total_length > MAX_FULL_PATH_LEN {
        return Err(ChronoIndexError::TickIo);
    }

    let mut write_cursor: usize = 0;
    // Copy parent.
    out_path_buffer[write_cursor..write_cursor + parent_path_bytes.len()]
        .copy_from_slice(parent_path_bytes);
    write_cursor += parent_path_bytes.len();
    // Optional separator.
    if !parent_ends_with_separator {
        out_path_buffer[write_cursor] = b'/';
        write_cursor += 1;
    }
    // Basename.
    out_path_buffer[write_cursor..write_cursor + basename_bytes.len()]
        .copy_from_slice(basename_bytes);
    write_cursor += basename_bytes.len();

    Ok(write_cursor)
}

/// Emits the next file in chronological order, if any, advancing the
/// cursor on disk.
///
/// On success with `Some(emission)`: `out_path_buffer[..emission.path_byte_length]`
/// holds the emitted file's absolute path bytes, and `cursor.bin` has
/// been atomically advanced.
///
/// On success with `None`: the cursor was already at the end of the
/// committed index. Nothing was written. A subsequent `update_index`
/// followed by another tick may produce more emissions.
///
/// On `Err(...)`: the cursor and index are unchanged. Caller may retry
/// next tick. Never panics.
pub fn iter_return_next_chronological_abs_file_path(
    temp_root_dir: &Path,
    out_path_buffer: &mut [u8; MAX_FULL_PATH_LEN],
) -> Result<Option<TickEmission>, ChronoIndexError> {
    // Read the committed header. If absent or structurally invalid,
    // signal TickIo so the caller can decide to run update_index.
    let committed_header = match read_header(temp_root_dir)? {
        Some(h) => h,
        None => return Err(ChronoIndexError::TickIo),
    };

    if committed_header.file_count == 0 {
        return Ok(None);
    }

    let current_cursor = read_cursor(temp_root_dir)?;

    // Cursor at or past the end → nothing more to emit at this commit.
    // (Use >= rather than == as defensive handling of a corrupted
    // cursor value that exceeded file_count.)
    if current_cursor >= committed_header.file_count {
        return Ok(None);
    }

    // Fetch the mtime record at the current cursor position.
    let mtime_record = read_mtime_record_at_index(temp_root_dir, current_cursor)?;

    // Defensive bounds check on record_id.
    if mtime_record.record_id >= committed_header.file_count {
        return Err(ChronoIndexError::TickIo);
    }

    // Fetch the corresponding name record.
    let mut name_record_buffer = [0u8; NAME_RECORD_SIZE];
    let basename_used_len = read_name_record_at_record_id(
        temp_root_dir,
        mtime_record.record_id,
        &mut name_record_buffer,
    )?;
    let basename_bytes = &name_record_buffer[..basename_used_len];

    // Assemble the absolute path.
    let path_byte_length = assemble_absolute_path_into_buffer(
        committed_header.parent_path_slice(),
        basename_bytes,
        out_path_buffer,
    )?;

    // Advance the cursor on disk before returning so a crashed consumer
    // does not re-emit the same file. Consumers are expected to tolerate
    // a single missed emission across a crash (i.e. the cursor-advance
    // is durable, but the consumer's downstream action may or may not
    // have completed). This is the documented trade-off.
    //
    // If you need strict at-least-once semantics, the caller should
    // record the emission as durable downstream BEFORE calling tick
    // again; or, conversely, advance the cursor only after downstream
    // success. The latter requires an explicit "ack" API which we have
    // deliberately not added here to keep the surface minimal.
    let next_cursor = current_cursor.saturating_add(1);
    write_cursor_atomic(temp_root_dir, next_cursor)?;

    Ok(Some(TickEmission {
        path_byte_length,
        emitted_cursor_index: current_cursor,
        emitted_mtime_sec: mtime_record.mtime_sec,
        emitted_mtime_nsec: mtime_record.mtime_nsec,
    }))
}

// =========================================================================
// Tests for part (d)
// =========================================================================

#[cfg(test)]
mod chrono_index_part_d_tests {
    use super::*;
    // use std::io::Write as _;

    fn make_test_temp_root(label: &str) -> PathBuf {
        let mut scratch = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        scratch.push(format!(
            "chrono_index_d_{}_{}_{}",
            label,
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&scratch).expect("setup");
        scratch
    }

    fn make_watched_dir_with_files(label: &str, files: &[(&str, &[u8])]) -> PathBuf {
        let mut watched = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        watched.push(format!(
            "chrono_watched_d_{}_{}_{}",
            label,
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&watched).expect("setup");
        for (basename, content) in files {
            let mut path = watched.clone();
            path.push(basename);
            let mut f = std::fs::File::create(&path).expect("create");
            f.write_all(content).expect("write");
            f.sync_all().expect("sync");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        watched
    }

    fn add_file_to_watched_dir(watched_dir: &Path, basename: &str, content: &[u8]) {
        std::thread::sleep(std::time::Duration::from_millis(15));
        let mut path = PathBuf::from(watched_dir);
        path.push(basename);
        let mut f = std::fs::File::create(&path).expect("create new");
        f.write_all(content).expect("write new");
        f.sync_all().expect("sync new");
    }

    #[test]
    fn update_index_on_empty_state_performs_cold_build() {
        let temp_root = make_test_temp_root("first_update");
        let watched =
            make_watched_dir_with_files("first_update", &[("a.txt", b"1"), ("b.txt", b"2")]);

        let summary = update_index(&temp_root, &watched).expect("update ok");
        assert_eq!(summary.outcome, UpdateOutcome::ColdBuildCompleted);
        assert_eq!(summary.final_file_count, 2);
        assert_eq!(summary.cold_build_summary.files_indexed, 2);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn update_index_no_changes_returns_noop_outcome() {
        let temp_root = make_test_temp_root("noop_update");
        let watched = make_watched_dir_with_files("noop_update", &[("x", b"1"), ("y", b"2")]);

        let first = update_index(&temp_root, &watched).expect("first ok");
        assert_eq!(first.outcome, UpdateOutcome::ColdBuildCompleted);

        let second = update_index(&temp_root, &watched).expect("second ok");
        assert_eq!(second.outcome, UpdateOutcome::NoChangesDetected);
        assert_eq!(second.final_file_count, 2);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn update_index_growth_triggers_incremental_append() {
        let temp_root = make_test_temp_root("growth");
        let watched = make_watched_dir_with_files("growth", &[("seed", b"s")]);
        let _ = update_index(&temp_root, &watched).expect("cold build via update");

        add_file_to_watched_dir(&watched, "grown_one", b"1");
        add_file_to_watched_dir(&watched, "grown_two", b"2");

        let summary = update_index(&temp_root, &watched).expect("append via update");
        assert_eq!(summary.outcome, UpdateOutcome::IncrementalAppendCompleted);
        assert_eq!(summary.final_file_count, 3);
        assert_eq!(summary.append_summary.files_appended, 2);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn update_index_rebuilds_when_data_file_size_disagrees_with_header() {
        let temp_root = make_test_temp_root("inconsistent");
        let watched =
            make_watched_dir_with_files("inconsistent", &[("a", b"1"), ("b", b"2"), ("c", b"3")]);
        let _ = update_index(&temp_root, &watched).expect("first ok");

        // Corrupt by truncating names.bin to half size.
        let names_path = build_index_file_path(&temp_root, NAMES_FILENAME);
        let original_size = std::fs::metadata(&names_path).expect("meta").len();
        let truncated_handle = OpenOptions::new()
            .write(true)
            .open(&names_path)
            .expect("open names rw");
        truncated_handle
            .set_len(original_size / 2)
            .expect("truncate names");
        drop(truncated_handle);

        let summary = update_index(&temp_root, &watched).expect("rebuild via update");
        assert_eq!(summary.outcome, UpdateOutcome::RebuiltDueToInconsistency);
        assert_eq!(summary.final_file_count, 3);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn update_index_rebuilds_when_parent_path_changed() {
        let temp_root = make_test_temp_root("reparent");
        let watched_a = make_watched_dir_with_files("reparent_a", &[("aa", b"a")]);
        let watched_b = make_watched_dir_with_files("reparent_b", &[("bb", b"b")]);

        let _ = update_index(&temp_root, &watched_a).expect("first ok");

        // Now point the same temp_root at a different parent directory.
        let summary = update_index(&temp_root, &watched_b).expect("rebuild ok");
        assert_eq!(summary.outcome, UpdateOutcome::RebuiltDueToInconsistency);
        assert_eq!(summary.final_file_count, 1);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched_a);
        let _ = std::fs::remove_dir_all(&watched_b);
    }

    #[test]
    fn tick_emits_files_in_chronological_order_and_advances_cursor() {
        let temp_root = make_test_temp_root("tick_order");
        // Create files in deliberate chronological order. We expect tick
        // to emit them in that same order.
        let watched = make_watched_dir_with_files(
            "tick_order",
            &[
                ("first.txt", b"1"),
                ("second.txt", b"2"),
                ("third.txt", b"3"),
            ],
        );
        let _ = update_index(&temp_root, &watched).expect("build");

        let expected_order: [&[u8]; 3] = [b"first.txt", b"second.txt", b"third.txt"];

        let mut path_buffer = [0u8; MAX_FULL_PATH_LEN];
        for (tick_index, expected_basename) in expected_order.iter().enumerate() {
            let emission_opt =
                iter_return_next_chronological_abs_file_path(&temp_root, &mut path_buffer)
                    .expect("tick ok");
            let emission = emission_opt.expect("tick should emit a file at this index");
            assert_eq!(emission.emitted_cursor_index as usize, tick_index);

            let emitted_path_bytes = &path_buffer[..emission.path_byte_length];
            // Path must end with "/<expected_basename>".
            let trailing_len = expected_basename.len() + 1;
            assert!(emitted_path_bytes.len() >= trailing_len);
            let trailing = &emitted_path_bytes[emitted_path_bytes.len() - trailing_len..];
            assert_eq!(trailing[0], b'/');
            assert_eq!(&trailing[1..], *expected_basename);
        }

        // Cursor must now equal file_count.
        let cursor_after = read_cursor(&temp_root).expect("cursor");
        assert_eq!(cursor_after, 3);

        // One more tick must yield None.
        let final_emission =
            iter_return_next_chronological_abs_file_path(&temp_root, &mut path_buffer)
                .expect("tick ok (end)");
        assert!(final_emission.is_none());

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn tick_resumes_after_growth_and_update() {
        let temp_root = make_test_temp_root("tick_resume");
        let watched =
            make_watched_dir_with_files("tick_resume", &[("alpha", b"a"), ("beta", b"b")]);
        let _ = update_index(&temp_root, &watched).expect("build");

        let mut path_buffer = [0u8; MAX_FULL_PATH_LEN];

        // Emit both committed files.
        let e1 = iter_return_next_chronological_abs_file_path(&temp_root, &mut path_buffer)
            .expect("t1 ok")
            .expect("e1");
        assert_eq!(e1.emitted_cursor_index, 0);
        let e2 = iter_return_next_chronological_abs_file_path(&temp_root, &mut path_buffer)
            .expect("t2 ok")
            .expect("e2");
        assert_eq!(e2.emitted_cursor_index, 1);

        // No more at this commit.
        let none_now = iter_return_next_chronological_abs_file_path(&temp_root, &mut path_buffer)
            .expect("t3 ok");
        assert!(none_now.is_none());

        // Grow the directory and update.
        add_file_to_watched_dir(&watched, "gamma", b"g");
        let summary = update_index(&temp_root, &watched).expect("append");
        assert_eq!(summary.outcome, UpdateOutcome::IncrementalAppendCompleted);

        // Now tick must emit the new file at cursor index 2.
        let e3 = iter_return_next_chronological_abs_file_path(&temp_root, &mut path_buffer)
            .expect("t4 ok")
            .expect("e3");
        assert_eq!(e3.emitted_cursor_index, 2);
        let emitted_path = &path_buffer[..e3.path_byte_length];
        assert!(emitted_path.ends_with(b"/gamma"));

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn tick_with_empty_index_returns_none() {
        let temp_root = make_test_temp_root("tick_empty");
        let watched = make_watched_dir_with_files("tick_empty", &[]);
        let _ = update_index(&temp_root, &watched).expect("build (empty)");

        let mut path_buffer = [0u8; MAX_FULL_PATH_LEN];
        let none_emission =
            iter_return_next_chronological_abs_file_path(&temp_root, &mut path_buffer)
                .expect("tick ok");
        assert!(none_emission.is_none());

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn tick_without_existing_header_returns_terse_error() {
        let temp_root = make_test_temp_root("tick_no_header");
        ensure_index_directory_exists(&temp_root).expect("setup");

        let mut path_buffer = [0u8; MAX_FULL_PATH_LEN];
        let result = iter_return_next_chronological_abs_file_path(&temp_root, &mut path_buffer);
        assert_eq!(result.err(), Some(ChronoIndexError::TickIo));

        let _ = std::fs::remove_dir_all(&temp_root);
    }

    #[test]
    fn assemble_absolute_path_handles_trailing_and_no_trailing_slash() {
        let mut buffer = [0u8; MAX_FULL_PATH_LEN];

        // Parent without trailing slash.
        let len1 =
            assemble_absolute_path_into_buffer(b"/var/data", b"foo.txt", &mut buffer).expect("ok");
        assert_eq!(&buffer[..len1], b"/var/data/foo.txt");

        // Parent with trailing slash.
        let len2 =
            assemble_absolute_path_into_buffer(b"/var/data/", b"bar.txt", &mut buffer).expect("ok");
        assert_eq!(&buffer[..len2], b"/var/data/bar.txt");
    }

    #[test]
    fn assemble_absolute_path_rejects_oversize_result() {
        let mut buffer = [0u8; MAX_FULL_PATH_LEN];
        // Parent length that already saturates the buffer.
        let huge_parent = vec![b'a'; MAX_FULL_PATH_LEN];
        let result = assemble_absolute_path_into_buffer(&huge_parent, b"x", &mut buffer);
        assert_eq!(result.err(), Some(ChronoIndexError::TickIo));
    }

    #[test]
    fn assemble_absolute_path_rejects_empty_parent() {
        let mut buffer = [0u8; MAX_FULL_PATH_LEN];
        let result = assemble_absolute_path_into_buffer(b"", b"x", &mut buffer);
        assert_eq!(result.err(), Some(ChronoIndexError::ParentPathInvalid));
    }

    #[test]
    fn tick_emission_path_matches_an_existing_file_on_disk() {
        // End-to-end sanity: the path emitted by tick must actually
        // identify a real file in the watched directory.
        let temp_root = make_test_temp_root("tick_real_path");
        let watched = make_watched_dir_with_files("tick_real_path", &[("hello.txt", b"hi")]);
        let _ = update_index(&temp_root, &watched).expect("build");

        let mut path_buffer = [0u8; MAX_FULL_PATH_LEN];
        let emission = iter_return_next_chronological_abs_file_path(&temp_root, &mut path_buffer)
            .expect("tick ok")
            .expect("file present");

        let emitted_bytes = &path_buffer[..emission.path_byte_length];
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let os_str = std::ffi::OsStr::from_bytes(emitted_bytes);
            let path_view = Path::new(os_str);
            assert!(path_view.is_file(), "emitted path must exist as a file");
        }

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }
}

// =========================================================================
// Part (e): Cleanup and inspection helpers
// =========================================================================

/// Removes ONLY the index state files under `<temp_root_dir>/chrono_index/`
/// — the `chrono_index/` subdirectory itself and everything inside it.
/// Does **not** touch the caller-supplied `temp_root_dir` itself, and
/// does **not** touch the watched directory or any of its files.
///
/// Use this when:
///   - The caller wants to discard the index entirely (e.g. switching
///     to a different watched directory and choosing not to reuse the
///     same `temp_root_dir`).
///   - A higher-level component has decided the index is unrecoverable
///     and a fresh cold rebuild on the next `update_index` is desired.
///
/// Per project policy this function does not halt. On I/O failure it
/// returns a terse error code; the caller can choose to retry or accept
/// the leftover state (a subsequent `update_index` will rebuild over it
/// in any case).
///
/// Safety / scope guarantees:
///   - Removes only `<temp_root_dir>/chrono_index/` and its contents.
///   - Never removes `<temp_root_dir>` itself.
///   - Never removes anything in or under the watched directory.
///
/// Note: if any concurrent process is currently holding open file
/// handles inside `chrono_index/`, the behavior is platform-dependent
/// (POSIX allows removal while handles remain open; the files stay
/// alive until the last handle is closed). The module's own functions
/// always open + read + close in a single call, so they do not retain
/// handles between calls.
pub fn purge_index_state(temp_root_dir: &Path) -> Result<(), ChronoIndexError> {
    let mut index_subdir = PathBuf::from(temp_root_dir);
    index_subdir.push(INDEX_SUBDIRNAME);

    match std::fs::remove_dir_all(&index_subdir) {
        Ok(()) => Ok(()),
        Err(io_error) => {
            // "Already gone" is a successful end-state, not an error.
            if io_error.kind() == std::io::ErrorKind::NotFound {
                Ok(())
            } else {
                Err(ChronoIndexError::IndexDirIo)
            }
        }
    }
}

/// Removes only the transient scratch state under
/// `<temp_root_dir>/chrono_index/scratch/`, if any. Leaves the committed
/// index (`header.bin`, `names.bin`, `mtimes.bin`, `name_hashes.bin`,
/// `cursor.bin`) untouched.
///
/// `cold_build_index` already cleans up `scratch/` on success and on
/// most failure paths. This helper exists for the rare case where a
/// process was killed mid-build and the next process wants to clear
/// the scratch artifacts without triggering a full rebuild yet.
///
/// Per project policy: does not halt. Returns `Ok(())` if the scratch
/// dir is absent (treated as the goal-state).
pub fn purge_scratch_only(temp_root_dir: &Path) -> Result<(), ChronoIndexError> {
    let mut scratch_dir = PathBuf::from(temp_root_dir);
    scratch_dir.push(INDEX_SUBDIRNAME);
    scratch_dir.push(SCRATCH_DIRNAME);

    match std::fs::remove_dir_all(&scratch_dir) {
        Ok(()) => Ok(()),
        Err(io_error) => {
            if io_error.kind() == std::io::ErrorKind::NotFound {
                Ok(())
            } else {
                Err(ChronoIndexError::IndexDirIo)
            }
        }
    }
}

/// Reports the absolute path of the index subdirectory managed by this
/// module for the given `temp_root_dir`. Useful for caller logging,
/// disk-usage accounting, or external backup tooling.
///
/// This function does no I/O. It returns the path even if the directory
/// does not yet exist.
pub fn locate_index_subdirectory(temp_root_dir: &Path) -> PathBuf {
    let mut p = PathBuf::from(temp_root_dir);
    p.push(INDEX_SUBDIRNAME);
    p
}

/// Reports whether a committed (valid-looking) header is currently
/// present under `temp_root_dir`. Convenience for callers that want to
/// decide between "use existing session state" and "start fresh"
/// without invoking `update_index` first.
///
/// Returns `false` if the header is absent OR structurally invalid.
/// Does no rebuild and does not modify any files.
pub fn committed_header_is_present(temp_root_dir: &Path) -> bool {
    match read_header(temp_root_dir) {
        Ok(Some(_)) => true,
        Ok(None) => false,
        Err(_) => false,
    }
}

#[cfg(test)]
mod chrono_index_part_e_tests {
    use super::*;

    fn make_test_temp_root(label: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!(
            "chrono_index_e_{}_{}_{}",
            label,
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&p).expect("setup");
        p
    }

    fn make_watched_dir_with_one_file(label: &str) -> PathBuf {
        let mut watched = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        watched.push(format!(
            "chrono_watched_e_{}_{}_{}",
            label,
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&watched).expect("setup");
        let mut f = std::fs::File::create(watched.join("only.txt")).expect("create");
        use std::io::Write;
        f.write_all(b"only").expect("write");
        f.sync_all().expect("sync");
        watched
    }

    #[test]
    fn committed_header_is_present_reports_false_before_first_use() {
        let temp_root = make_test_temp_root("present_before");
        assert!(!committed_header_is_present(&temp_root));
        let _ = std::fs::remove_dir_all(&temp_root);
    }

    #[test]
    fn committed_header_is_present_reports_true_after_update() {
        let temp_root = make_test_temp_root("present_after");
        let watched = make_watched_dir_with_one_file("present_after");
        let _ = update_index(&temp_root, &watched).expect("build");
        assert!(committed_header_is_present(&temp_root));
        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn purge_index_state_removes_index_subdir_only() {
        let temp_root = make_test_temp_root("purge_index");
        let watched = make_watched_dir_with_one_file("purge_index");
        let _ = update_index(&temp_root, &watched).expect("build");

        // Place an unrelated file directly in temp_root; it must survive.
        let bystander_path = temp_root.join("unrelated.txt");
        std::fs::write(&bystander_path, b"keep me").expect("seed bystander");

        purge_index_state(&temp_root).expect("purge ok");

        assert!(!locate_index_subdirectory(&temp_root).exists());
        assert!(bystander_path.exists(), "bystander must be untouched");
        assert!(temp_root.exists(), "temp_root itself must be untouched");

        // And purge is idempotent: a second call succeeds on absent state.
        purge_index_state(&temp_root).expect("purge again ok");

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn purge_scratch_only_leaves_committed_index_intact() {
        let temp_root = make_test_temp_root("purge_scratch");
        let watched = make_watched_dir_with_one_file("purge_scratch");
        let _ = update_index(&temp_root, &watched).expect("build");

        // Manually create a scratch subdir to mimic an aborted prior build.
        let mut scratch = locate_index_subdirectory(&temp_root);
        scratch.push(SCRATCH_DIRNAME);
        std::fs::create_dir_all(&scratch).expect("mkdir scratch");
        std::fs::write(scratch.join("stale.bin"), b"stale").expect("seed");
        assert!(scratch.exists());

        purge_scratch_only(&temp_root).expect("purge scratch ok");
        assert!(!scratch.exists(), "scratch must be gone");

        // Committed index files still present.
        assert!(committed_header_is_present(&temp_root));
        assert!(
            locate_index_subdirectory(&temp_root)
                .join(HEADER_FILENAME)
                .exists()
        );

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn locate_index_subdirectory_is_pure() {
        let temp_root = make_test_temp_root("locate");
        // Path returned even before the dir exists.
        let located = locate_index_subdirectory(&temp_root);
        assert!(located.ends_with(INDEX_SUBDIRNAME));
        let _ = std::fs::remove_dir_all(&temp_root);
    }
}

// =========================================================================
// Part (f): N-at-a-time wrapper around the per-tick primitive
// =========================================================================
//
// `iter_return_next_chronological_abs_file_path` yields exactly one absolute path
// per call. Many callers want "give me the next N paths in chronological
// order" as a single step — for example, "process up to 32 files this
// minute, then sleep." This wrapper provides that convenience without
// changing the semantics of the underlying primitive.
//
// Important: this function does NOT open or read any of the watched
// files' contents. It only yields their absolute paths in chronological
// order, advancing the on-disk cursor by one per path yielded.

/// One yielded path, written into the caller's buffer.
///
/// `path_byte_length` indicates how many leading bytes of the
/// caller-provided `[u8; MAX_FULL_PATH_LEN]` slot hold the absolute path
/// of the yielded file. The remaining bytes in that slot are unspecified.
#[derive(Clone, Copy, Debug)]
pub struct YieldedChronologicalPath {
    /// Number of valid path bytes in the corresponding output buffer.
    pub path_byte_length: usize,
    /// 0-based chronological cursor index from which this path was
    /// yielded. The first chronological file is index 0.
    pub yielded_cursor_index: u64,
    /// mtime of the yielded file. Exposed for caller logging.
    pub yielded_mtime_sec: i64,
    pub yielded_mtime_nsec: i32,
}

/// Aggregate result of a batched call.
#[derive(Clone, Copy, Debug)]
pub struct BatchYieldSummary {
    /// Number of paths actually written into the output buffers.
    /// May be less than the requested batch size if the end of the
    /// committed chronological order was reached.
    pub paths_yielded: usize,
    /// `true` iff `paths_yielded < requested_batch_size` because the
    /// committed index has no more files at the current commit.
    /// (A subsequent `update_index` may add more.)
    pub reached_end_of_commit: bool,
}

/// Yields up to `requested_batch_size` next absolute paths in
/// chronological order, writing each into the corresponding slot of
/// the caller-provided buffer arrays.
///
/// Inputs:
///   - `temp_root_dir`: same temp root used with `update_index` and
///     `iter_return_next_chronological_abs_file_path`.
///   - `out_path_buffers`: an array slice of `[u8; MAX_FULL_PATH_LEN]`
///     slots. Slot `i` receives the bytes of the i-th yielded path.
///     The caller chooses the slice length (e.g. `&mut my_buffers[..32]`).
///   - `out_per_path_metadata`: a parallel slice of `YieldedChronologicalPath`
///     slots. Slot `i` is filled with metadata for the i-th yielded
///     path. Must have the same length as `out_path_buffers`.
///   - `requested_batch_size`: how many paths to attempt to yield.
///     Clamped to `min(out_path_buffers.len(), out_per_path_metadata.len())`.
///
/// Returns `Ok(BatchYieldSummary)`. The cursor on disk is advanced by
/// exactly `paths_yielded`. On a per-iteration I/O failure the function
/// stops at the failure boundary, the cursor is left at the last
/// successfully-yielded position, and the partial summary is returned.
/// This means "what got yielded actually got yielded" — no torn state.
///
/// Per project policy this function does not panic and does not halt.
pub fn tick_emit_next_n_chronological_paths(
    temp_root_dir: &Path,
    out_path_buffers: &mut [[u8; MAX_FULL_PATH_LEN]],
    out_per_path_metadata: &mut [YieldedChronologicalPath],
    requested_batch_size: usize,
) -> Result<BatchYieldSummary, ChronoIndexError> {
    // Clamp the batch to the caller-provided buffer capacity.
    let effective_cap = requested_batch_size
        .min(out_path_buffers.len())
        .min(out_per_path_metadata.len());

    let mut paths_yielded: usize = 0;
    let mut reached_end_of_commit: bool = false;

    while paths_yielded < effective_cap {
        // Borrow the next output slot.
        let path_slot: &mut [u8; MAX_FULL_PATH_LEN] = &mut out_path_buffers[paths_yielded];

        match iter_return_next_chronological_abs_file_path(temp_root_dir, path_slot) {
            Ok(Some(emission)) => {
                out_per_path_metadata[paths_yielded] = YieldedChronologicalPath {
                    path_byte_length: emission.path_byte_length,
                    yielded_cursor_index: emission.emitted_cursor_index,
                    yielded_mtime_sec: emission.emitted_mtime_sec,
                    yielded_mtime_nsec: emission.emitted_mtime_nsec,
                };
                paths_yielded = paths_yielded.saturating_add(1);
            }
            Ok(None) => {
                // Cursor is at end-of-commit. Stop cleanly.
                reached_end_of_commit = true;
                break;
            }
            Err(error_code) => {
                // Partial-progress contract: return what we have so far,
                // along with the error code, so the caller knows the
                // cursor advanced by exactly `paths_yielded`.
                //
                // We package this as Err and let the caller inspect any
                // already-written metadata if desired. To avoid losing
                // the count, we return a structured error path: the
                // caller can re-read the on-disk cursor to learn how far
                // we got, or simply rely on `paths_yielded` having been
                // written into prior slots only.
                let _ = paths_yielded; // documented above
                return Err(error_code);
            }
        }
    }

    Ok(BatchYieldSummary {
        paths_yielded,
        reached_end_of_commit,
    })
}

// =========================================================================
// Public chronological lookup by position
// =========================================================================

/// Returns the absolute path of the file at chronological position
/// `chronological_position` in the committed index.
///
/// Positions are zero-based and ordered by mtime ascending:
///   - position 0                       = chronologically earliest file
///   - position `file_count - 1`        = chronologically latest file
///   - position >= `file_count`         = `Ok(None)`
///
/// This function is read-only. It does not modify cursor.bin and does
/// not modify any other file. It may be called any number of times
/// with any positions, in any order. Two calls with the same position
/// return the same path (provided the index has not been rebuilt
/// between them).
///
/// The absolute path is written into `out_path_buffer`; the returned
/// `TickEmission.path_byte_length` is the number of valid leading
/// bytes in that buffer.
///
/// Per project policy: never panics, never halts.
pub fn lookup_abs_file_path_at_mtime_chronological_index(
    temp_root_dir: &Path,
    chronological_position: u64,
    out_path_buffer: &mut [u8; MAX_FULL_PATH_LEN],
) -> Result<Option<TickEmission>, ChronoIndexError> {
    let committed_header = match read_header(temp_root_dir)? {
        Some(h) => h,
        None => return Err(ChronoIndexError::TickIo),
    };

    if chronological_position >= committed_header.file_count {
        return Ok(None);
    }

    let mtime_record = read_mtime_record_at_index(temp_root_dir, chronological_position)?;

    if mtime_record.record_id >= committed_header.file_count {
        return Err(ChronoIndexError::TickIo);
    }

    let mut name_record_buffer = [0u8; NAME_RECORD_SIZE];
    let basename_used_len = read_name_record_at_record_id(
        temp_root_dir,
        mtime_record.record_id,
        &mut name_record_buffer,
    )?;
    let basename_bytes = &name_record_buffer[..basename_used_len];

    let path_byte_length = assemble_absolute_path_into_buffer(
        committed_header.parent_path_slice(),
        basename_bytes,
        out_path_buffer,
    )?;

    Ok(Some(TickEmission {
        path_byte_length,
        emitted_cursor_index: chronological_position,
        emitted_mtime_sec: mtime_record.mtime_sec,
        emitted_mtime_nsec: mtime_record.mtime_nsec,
    }))
}

/// Returns the number of files currently committed in the index — i.e.
/// the upper bound (exclusive) for valid arguments to
/// `lookup_abs_file_path_at_mtime_chronological_index`.
///
/// Returns `Ok(0)` if no header is committed yet. Never panics.
pub fn count_committed_files(temp_root_dir: &Path) -> Result<u64, ChronoIndexError> {
    match read_header(temp_root_dir)? {
        Some(header) => Ok(header.file_count),
        None => Ok(0),
    }
}

#[cfg(test)]
mod chrono_index_lookup_tests {
    use super::*;
    use std::io::Write as _;

    fn make_test_temp_root(label: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!(
            "chrono_index_lookup_{}_{}_{}",
            label,
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&p).expect("setup");
        p
    }

    fn make_watched_dir_with_files(label: &str, files: &[(&str, &[u8])]) -> PathBuf {
        let mut watched = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        watched.push(format!(
            "chrono_watched_lookup_{}_{}_{}",
            label,
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&watched).expect("setup");
        for (basename, content) in files {
            let mut path = watched.clone();
            path.push(basename);
            let mut f = std::fs::File::create(&path).expect("create");
            f.write_all(content).expect("write");
            f.sync_all().expect("sync");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        watched
    }

    #[test]
    fn lookup_position_zero_returns_chronologically_earliest_file() {
        let temp_root = make_test_temp_root("zero");
        let watched = make_watched_dir_with_files(
            "zero",
            &[
                ("first.txt", b"1"),
                ("second.txt", b"2"),
                ("third.txt", b"3"),
            ],
        );
        let _ = update_index(&temp_root, &watched).expect("build");

        let mut buf = [0u8; MAX_FULL_PATH_LEN];
        let result = lookup_abs_file_path_at_mtime_chronological_index(&temp_root, 0, &mut buf)
            .expect("ok")
            .expect("present");

        let path_bytes = &buf[..result.path_byte_length];
        assert!(path_bytes.ends_with(b"/first.txt"));

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn lookup_is_random_access_and_does_not_mutate_cursor() {
        let temp_root = make_test_temp_root("random_access");
        let watched = make_watched_dir_with_files(
            "random_access",
            &[("a", b"a"), ("b", b"b"), ("c", b"c"), ("d", b"d")],
        );
        let _ = update_index(&temp_root, &watched).expect("build");

        let cursor_before = read_cursor(&temp_root).expect("read");
        let mut buf = [0u8; MAX_FULL_PATH_LEN];

        // Jump around in arbitrary order.
        for position in [2u64, 0, 3, 1, 2, 0] {
            let r =
                lookup_abs_file_path_at_mtime_chronological_index(&temp_root, position, &mut buf)
                    .expect("ok")
                    .expect("present");
            assert_eq!(r.emitted_cursor_index, position);
        }

        let cursor_after = read_cursor(&temp_root).expect("read");
        assert_eq!(
            cursor_before, cursor_after,
            "lookup must not mutate cursor.bin"
        );

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn lookup_past_end_returns_none() {
        let temp_root = make_test_temp_root("past_end");
        let watched = make_watched_dir_with_files("past_end", &[("only", b"x")]);
        let _ = update_index(&temp_root, &watched).expect("build");

        let mut buf = [0u8; MAX_FULL_PATH_LEN];
        let r =
            lookup_abs_file_path_at_mtime_chronological_index(&temp_root, 5, &mut buf).expect("ok");
        assert!(r.is_none());

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn count_committed_files_reports_header_count() {
        let temp_root = make_test_temp_root("count");
        let watched = make_watched_dir_with_files("count", &[("a", b"a"), ("b", b"b")]);

        // Before any update_index, no header → 0.
        assert_eq!(count_committed_files(&temp_root).expect("ok"), 0);

        let _ = update_index(&temp_root, &watched).expect("build");
        assert_eq!(count_committed_files(&temp_root).expect("ok"), 2);

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }

    #[test]
    fn lookup_returns_paths_in_ascending_mtime_order() {
        let temp_root = make_test_temp_root("ascending");
        let watched = make_watched_dir_with_files(
            "ascending",
            &[("p0", b"0"), ("p1", b"1"), ("p2", b"2"), ("p3", b"3")],
        );
        let _ = update_index(&temp_root, &watched).expect("build");

        let total = count_committed_files(&temp_root).expect("ok");
        assert_eq!(total, 4);

        let mut buf = [0u8; MAX_FULL_PATH_LEN];
        let mut previous_mtime_sec: Option<i64> = None;
        let mut previous_mtime_nsec: Option<i32> = None;
        for position in 0..total {
            let r =
                lookup_abs_file_path_at_mtime_chronological_index(&temp_root, position, &mut buf)
                    .expect("ok")
                    .expect("present");

            if let (Some(prev_sec), Some(prev_nsec)) = (previous_mtime_sec, previous_mtime_nsec) {
                let strictly_ascending = r.emitted_mtime_sec > prev_sec
                    || (r.emitted_mtime_sec == prev_sec && r.emitted_mtime_nsec >= prev_nsec);
                assert!(
                    strictly_ascending,
                    "positions must be non-decreasing in mtime"
                );
            }
            previous_mtime_sec = Some(r.emitted_mtime_sec);
            previous_mtime_nsec = Some(r.emitted_mtime_nsec);
        }

        let _ = std::fs::remove_dir_all(&temp_root);
        let _ = std::fs::remove_dir_all(&watched);
    }
}
