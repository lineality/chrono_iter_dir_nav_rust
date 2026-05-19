#### chrono_sort_dir_nav_rust

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

# Discussion

(note: this system tries to avoid using heap memory wherever possible, not always possible)

## Primary use-case in summary:
Reading files in a directory where files are not deleted but where different threads write files, and so possibly write files in a 'colliding sequence,' such that periodically checking the sequence is needed.
E.g. if a relatively delayed thread is creating a file but takes longer, that file may have an earlier mtime than a file added by another thread. E.g. if At time T your look at chrono-index file [4], after you do that, and potentially after you double-check that, another thread may finish creating what then becomes a different chrono-index file [4],
such that either:
A. for each "get 'next' file path" action to update state do a sequence-check (e.g. a rolling hash of file-id or file-name), or
B. a separate more periodic safety-check to trigger a rebuild, or just rebuild state.
there needs to be a process that does not assume that file sequence cannot have edge case issues

We do not want a case where the system is continually cycling and rebuilding multiple times a second, as in most casts several minutes (many tens of minutes pass (in some case hours pass, or days, or weeks) between any new file being added.

Could it be enough to base re-building on a length check? Probably not.

We should consider a few options and or tools to use in combination:
1. A length check
2. An OS-returned-order (unpredictabale) rolling hash (salted or not)
3. boolean test: change in Chrono-sorted-hash-to-index-N


More specifically:
the game-state needs to ask:
- I want to see the 'next' white move, the next black-move,
and it needs to know if the 'next' move file is now (by edge case) earlier in the chrono-queue.

### Tools:
- length check: a length check alone does not tell you if history retroactively changed, or if (rarely) a file was deleted and another added

- hash check: superficially a hash-check (e.g. pearson hash ongoing of every file in dir) will supplement a length check in saying if something changed, but like a length check a basic hash of all files alone does not tell you if history retroactively changed.

- sequence check: the idea of a sequence check is to sort by mtime and then do a hash check of all or up to N index. Due to edge cases a chronological sort is a useful tool here.


# Plan-A: simple brute force

1. every N-seconds (refresh rate)
Do an OS file-iter read, one by one, and keep A. and count, and B. a rolling pearson-hash of the file-id and or file-name. (avoiding heap use)
(question: is posix file-id strongly unique?)

(in most cases, no change, no action)


2. In case of any change to either (length or hash)
re-build game-state from scratch (running through all moves)

note: the salted-hash is probably excessive
note: while files should not be deleted, the hash will catch that if it happens



The specific use case here is a chess game (average of 40 moves, plus some setup files) happening where each player-thread saves move-files independently.

The pace of doing periodic length checks is config-set per game and may range from 1 to N (e.g. 30 or 60) sec.

for the vast majority of directory-array length (new file checks) there will be no change, no new file added, and nothing to do, (no change in state).

And in an average 'game' there will be <=64 new-file-found updates, making it not-too-onerous to re-build state that many times over very rarely less than 10-min.

However, end-game scrambles (e.g. where kings chase each other for 60 moves rapidly) are a bottleneck worth looking at. Even if there are only 50 moves in the game, re-reading 50-files every second is not going to break a computer, but IF (if) there is sound way to avoid this it should be looked at.
E.g. checking for a 'history-re-write' (a retro-active change in the sequence of files) every second is much cheaper that reading every file in the directory (ever growing) every second.

(The worst cases are rare, but also the worst: e.g. 237-move Rapid-Chess game between Alexandra Kosteniuk and Laurent Fressinet in 2007. That would be re-reading 200+ files every second, with an ever-growing stack of files. Vs. Only re-building state IF there is a chronology-re-writing change, otherwise much more cheaply looking at 'the chronologically next' file.


# Plan-B. only rebuild whole state based on change in Chrono-sorted-hash-to-index-N

(Note: 'chronological' has edge cases where mtime may not be unique for files (though it is most of the time) so there needs to be some rule for "chronological" means mtime with tiebreak by filename. I think posix rules are that file-name within a dir must be strongly unique (note: file-id might (will) be recycled over time)

A critical chrono-sort function may be check_chronosort_hash_to_n(index_n, previous_identical_hash) -> <Result Bool>

If the past-timeline has not changed, no need to re-build state...maybe.
If past-timelines has changed, a full state rebuild is required.


- double check design and docs
- Q: rolling pearson hash check to make sure the past-composition hasn't changed
- maybe less-frequently do a history-check/rebuild
- note: iterate through and count owners vs. simple next item
e.g. assume there can be (somewhat rare) collision sequence issues
so that any one read may be one-off compared with the next read.

- possible safeguard: Doing a count before and after
e.g. would 1. count 2. 'next' 3. count
if both counts are the same, can there have been a collision in the middle?
It may depend on how delayed a thread can be.

# Conclusions:
1. Is a simple length check all that is ever needed? I think: no
2. Is a simple brute force constant re-read of all files too expensive: I think so.
3. Is a state-rebuild based on 'check_chronosort_hash_to_n() == false' reliable and sufficient? It think: yes
4. Is a state-rebuild based on 'check_chronosort_hash_to_n() == false' affordable in cost? It think: yes

#### Note:
For the main use-case, to safe memory, file-names are known to be short, (<32 char)
