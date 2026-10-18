// Seeded PRNG (mulberry32) with Python-random-like helpers.
// Seeds are deterministic within the PWA but do NOT match Python's output.
export class RNG {
  constructor(seed = 42) {
    this.s = (seed >>> 0) || 0x9e3779b9;
  }
  next() {
    this.s = (this.s + 0x6d2b79f5) | 0;
    let t = Math.imul(this.s ^ (this.s >>> 15), 1 | this.s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  // randrange(n) -> [0,n); randrange(a,b) -> [a,b)
  randrange(a, b) {
    if (b === undefined) { b = a; a = 0; }
    return a + Math.floor(this.next() * (b - a));
  }
  choice(arr) { return arr[this.randrange(arr.length)]; }
  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.randrange(i + 1);
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
  sample(arr, k) {
    const copy = [...arr];
    this.shuffle(copy);
    return copy.slice(0, k);
  }
}
