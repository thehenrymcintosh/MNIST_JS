import { FiniteIterator, Matrix } from "./types";

export function zip<T>(...arrs: T[][] ): T[][] {
  const len = arrs[0].length;
  arrs.forEach(arr => { if (arr.length !== len) throw new Error("Arrays must be same length to zip!"); });
  const zipped = [];
  for (let i = 0; i < len; i++) {
    const layer = arrs.map((arr) => arr[i]);
    zipped.push(layer);
  }
  return zipped
}

export function standard_normal() {
  let u = 0, v = 0;
  while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
  while(v === 0) v = Math.random();
  return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

export function arrayOf<T>(length: number, contents: (index: number) => T) {
  const arr = [];
  for (let i = 0; i < length; i++) {
    arr.push(contents(i));
  }
  return arr;
}

export function createMatrix(xLen: number, yLen: number, distribution: (x: number, y: number) => number ): Matrix {
  return arrayOf(yLen, (y) => arrayOf(xLen, (x) => distribution(x, y)))
}

export function msqErr(matrix: Matrix): number {
  let sqSum = 0;
  for (let i = 0; i < matrix.length; i++) {
    sqSum += matrix[i][0]**2;
  }
  return sqSum / matrix.length;
}

export class ArrayIterator<T> implements FiniteIterator<T> {
  private arr: T[];
  length: number;
  private position = 0;

  constructor(arr: T[]) {
    this.arr = arr;
    this.length = arr.length;
  }

  next() :IteratorResult<T> {
    const pos = this.position;
    if (pos >= this.length) return { value: null, done: true };
    this.position++;
    return {
      value: this.arr[pos],
      done: false,
    }
  }

  limit(lim: number) {
    if (lim < this.length) this.length = lim;
  }
}
