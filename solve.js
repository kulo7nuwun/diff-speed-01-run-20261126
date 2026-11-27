// A faster drop-in for makefun-diff 1.0.0: run(input) returns exactly make.fun's own patch
// of { old, new } as one JSON string, but computed on interned line ids with a typed-array
// Myers search instead of jsdiff's linked-list paths.
//
// The rule, restated from makefun-diff's own header: a text is cut after every "\n" (the
// only separator; "\r" is part of its line) and a last line with no "\n" is a line too. The
// common prefix of lines is kept, then the common suffix of the lines left. Between them
// Myers's greedy search runs from the first lines: for d = 0, 1, 2, ... edits and each
// diagonal k = x - y from -d to d in steps of 2, the furthest point a path of d edits
// reaches on k is either one old line past the furthest (d-1)-point on diagonal k - 1 (a
// deletion) or one new line past the furthest (d-1)-point on diagonal k + 1 (an insertion),
// whichever lies further into the old text, and the deletion when both reach the same
// point; a move past the end of either text is no candidate, no edge diagonal is pruned,
// and from that point the path follows every run of lines equal in both as far as it
// goes. The first path to reach the ends of both texts is the script.
//
// This file reproduces that search exactly, with two changes of representation only:
//  * lines are interned to small integers (one Map per case), so the search and its
//    snakes compare integers instead of strings;
//  * the search keeps, for every d, the furthest x reached on each diagonal in an
//    Int32Array (x = old lines consumed, -1 = diagonal dead), and the script is
//    recovered by walking those rows backwards with the same tie-break the forward
//    search used. The forward step, the snake, the end test and the tie-break are
//    the same arithmetic makefun-diff does on its paths' oldPos values.
//
// The patch is then built the same way makefun-diff builds it: header
// "makefun-diff/1 old=N new=M hunks=H deleted=D inserted=I", hunk headers
// "@ os ol ns nl", "=" / "-" / "+" lines, deletions before insertions in each change
// run, up to 3 kept lines before and after each run, runs at most 6 kept lines apart
// sharing a hunk, a line with no trailing newline written with one plus a following
// "%" line, identical texts the header line alone.

/** The kept lines a hunk shows before and after each change run. */
const CONTEXT = 3;

/** Op kinds, as small integers. */
const KEEP = 0;
const DELETE = 1;
const INSERT = 2;

/**
 * Flag bit OR-ed onto a line id when the line is a text's last line and ends with no
 * "\n": the same words with and without their newline are different lines, so they
 * must intern to different ids. Ids below this bit are newline-terminated lines.
 */
const BARE = 0x40000000;

/**
 * A text's lines as interned ids, with the split kept for writing the patch.
 * parts is text.split('\n'); the line at index i < last is parts[i] + '\n' and the
 * line at index last (when it exists) is parts[last] with no newline.
 */
function tokenize(text, map) {
  const parts = text.split('\n');
  const last = parts.length - 1;
  const n = last + (parts[last] !== '' ? 1 : 0);
  const ids = new Int32Array(n);
  for (let i = 0; i < last; i++) {
    let id = map.get(parts[i]);
    if (id === undefined) {
      id = map.size;
      map.set(parts[i], id);
    }
    ids[i] = id;
  }
  if (parts[last] !== '') {
    let id = map.get(parts[last]);
    if (id === undefined) {
      id = map.size;
      map.set(parts[last], id);
    }
    ids[last] = id | BARE;
  }
  return { ids, parts, last };
}

/**
 * The runs of the edit script of oldIds into newIds: [kind, count] pairs in order,
 * kinds KEEP / DELETE / INSERT. Exactly makefun-diff's editRuns, same tie-break.
 *
 * The furthest x reached on each diagonal is kept in one flat Int32Array, reused
 * across calls: row d starts at index d * d (the rows' lengths 2d + 1 sum to d^2),
 * entry k + d holds x = old lines consumed, DEAD where that diagonal is dead. The
 * script is recovered by walking those rows backwards with the same tie-break.
 */
/** A value no real x reaches: x never exceeds the old text's line count. */
const DEAD = 1000000000;

let V = new Int32Array(1 << 16);

function ensureV(need) {
  if (need <= V.length) return;
  let cap = V.length;
  while (cap < need) cap *= 2;
  const next = new Int32Array(cap);
  next.set(V);
  V = next;
}

function editRuns(a, b) {
  const N = a.length;
  const M = b.length;
  // The end (N, M) sits on diagonal endK: only that diagonal can reach it.
  const endK = N - M;
  // d = 0: the snake from (0, 0) on diagonal 0.
  let x = 0;
  let y = 0;
  while (x < N && y < M && a[x] === b[y]) {
    x++;
    y++;
  }
  V[0] = x;
  if (x === N && y === M) return [];
  for (let d = 1; d <= N + M; d++) {
    ensureV((d + 1) * (d + 1));
    const base = d * d;
    const pbase = (d - 1) * (d - 1);
    let foundK = 0;
    let found = false;
    // k = -d: only the insertion from diagonal k + 1 can reach here.
    {
      const xp = V[pbase];
      if (xp + d - 1 >= M) V[base] = DEAD;
      else {
        let xv = xp;
        let yv = xv + d;
        while (xv < N && yv < M && a[xv] === b[yv]) {
          xv++;
          yv++;
        }
        V[base] = xv;
        if (endK === -d && xv === N) {
          foundK = -d;
          found = true;
        }
      }
    }
    if (!found) {
      for (let k = -d + 2, idx = 2; k <= d - 2; k += 2, idx += 2) {
        const xm = V[pbase + idx - 2];
        const xp = V[pbase + idx];
        const canRemove = xm < N;
        const canAdd = xp - k - 1 < M;
        if (!canRemove && !canAdd) {
          V[base + idx] = DEAD;
          continue;
        }
        // Deletion when further into the old text, and on a tie: the deletion's
        // x is xm + 1, the insertion's is xp, so deletion wins when xm + 1 >= xp.
        let xv = canRemove && (!canAdd || xm + 1 >= xp) ? xm + 1 : xp;
        let yv = xv - k;
        while (xv < N && yv < M && a[xv] === b[yv]) {
          xv++;
          yv++;
        }
        V[base + idx] = xv;
        if (k === endK && xv === N) {
          foundK = k;
          found = true;
          break;
        }
      }
    }
    if (!found) {
      // k = d: only the deletion from diagonal k - 1 can reach here.
      const xm = V[pbase + 2 * d - 2];
      if (xm >= N) V[base + 2 * d] = DEAD;
      else {
        let xv = xm + 1;
        let yv = xv - d;
        while (xv < N && yv < M && a[xv] === b[yv]) {
          xv++;
          yv++;
        }
        V[base + 2 * d] = xv;
        if (endK === d && xv === N) {
          foundK = d;
          found = true;
        }
      }
    }
    if (found) return backtrack(d, foundK, N, M, a, b);
  }
  throw new Error('makefun-diff: the search ended without reaching both ends');
}

/**
 * Recover the script from the stored rows, walking backwards from (N, M) on
 * diagonal kEnd after dEnd edits. At each step the edit is re-decided from row
 * d - 1 with the forward rule, so the path is the one the forward search took.
 */
function backtrack(dEnd, kEnd, N, M, a, b) {
  const rev = [];
  const push = (kind, count) => {
    const t = rev[rev.length - 1];
    if (t !== undefined && t[0] === kind) t[1] += count;
    else rev.push([kind, count]);
  };
  let x = N;
  let y = M;
  let k = kEnd;
  for (let d = dEnd; d >= 1; d--) {
    const pbase = (d - 1) * (d - 1);
    let del;
    let preX;
    let preY;
    if (k === -d) {
      del = false;
      preX = V[pbase];
      preY = preX - (k + 1);
    } else if (k === d) {
      del = true;
      preX = V[pbase + 2 * d - 2];
      preY = preX - (k - 1);
    } else {
      const xm = V[pbase + k + d - 2];
      const xp = V[pbase + k + d];
      const canRemove = xm < N;
      const canAdd = xp - k - 1 < M;
      if (canRemove && (!canAdd || xm + 1 >= xp)) {
        del = true;
        preX = xm;
        preY = xm - (k - 1);
      } else {
        del = false;
        preX = xp;
        preY = xp - (k + 1);
      }
    }
    const postX = del ? preX + 1 : preX;
    const postY = del ? preY : preY + 1;
    const snake = x - postX;
    if (snake > 0) push(KEEP, snake);
    push(del ? DELETE : INSERT, 1);
    x = preX;
    y = preY;
    k = del ? k - 1 : k + 1;
  }
  if (x > 0) push(KEEP, x);
  rev.reverse();
  return rev;
}

/** make.fun's own patch of input.old into input.new, as one string. */
export function run(input) {
  if (input === null || typeof input !== 'object' || typeof input.old !== 'string' || typeof input.new !== 'string') {
    throw new TypeError('makefun-diff: the input is { old, new }, two strings');
  }
  const map = new Map();
  const A = tokenize(input.old, map);
  const B = tokenize(input.new, map);
  const a = A.ids;
  const b = B.ids;
  const n = a.length;
  const m = b.length;
  // The common prefix, then the common suffix of what is left.
  let prefix = 0;
  while (prefix < n && prefix < m && a[prefix] === b[prefix]) prefix++;
  let suffix = 0;
  while (suffix < n - prefix && suffix < m - prefix && a[n - 1 - suffix] === b[m - 1 - suffix]) suffix++;
  const runs = [];
  if (prefix > 0) runs.push([KEEP, prefix]);
  for (const r of editRuns(a.subarray(prefix, n - suffix), b.subarray(prefix, m - suffix))) runs.push(r);
  if (suffix > 0) runs.push([KEEP, suffix]);

  // The change runs: each a stretch of deletions and insertions between two kept lines.
  const changes = [];
  let o = 0;
  let p = 0;
  let deleted = 0;
  let inserted = 0;
  for (const [kind, count] of runs) {
    if (kind === KEEP) {
      o += count;
      p += count;
      continue;
    }
    let c = changes[changes.length - 1];
    if (c === undefined || c.o + c.dels !== o || c.p + c.ins !== p) {
      c = { o, p, dels: 0, ins: 0 };
      changes.push(c);
    }
    if (kind === DELETE) {
      c.dels += count;
      o += count;
      deleted += count;
    } else {
      c.ins += count;
      p += count;
      inserted += count;
    }
  }

  // The hunks: change runs at most 2 * CONTEXT kept lines apart share one.
  const hunks = [];
  for (const c of changes) {
    const h = hunks[hunks.length - 1];
    const prev = h !== undefined ? h[h.length - 1] : undefined;
    if (prev !== undefined && c.o - (prev.o + prev.dels) <= 2 * CONTEXT) h.push(c);
    else hunks.push([c]);
  }
  const out = [`makefun-diff/1 old=${n} new=${m} hunks=${hunks.length} deleted=${deleted} inserted=${inserted}\n`];
  // A line as the patch writes it: its op, the line, a newline, and "%" on a line of
  // its own when the line had none. A line at index i < tok.last ends with "\n".
  const emit = (op, tok, i) => {
    if (i < tok.last) out.push(op + tok.parts[i] + '\n');
    else out.push(op + tok.parts[tok.last] + '\n%\n');
  };
  for (const h of hunks) {
    const first = h[0];
    const last = h[h.length - 1];
    const oldStart = Math.max(0, first.o - CONTEXT);
    const newStart = first.p - (first.o - oldStart);
    const after = Math.min(CONTEXT, n - (last.o + last.dels));
    const oldEnd = last.o + last.dels + after;
    const newEnd = last.p + last.ins + after;
    out.push(`@ ${oldStart + 1} ${oldEnd - oldStart} ${newStart + 1} ${newEnd - newStart}\n`);
    let at = oldStart;
    for (const c of h) {
      for (; at < c.o; at++) emit('=', A, at);
      for (let i = 0; i < c.dels; i++) emit('-', A, c.o + i);
      for (let i = 0; i < c.ins; i++) emit('+', B, c.p + i);
      at = c.o + c.dels;
    }
    for (; at < oldEnd; at++) emit('=', A, at);
  }
  return out.join('');
}
