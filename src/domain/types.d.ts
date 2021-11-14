export type Matrix = number[][];

export interface FiniteIterator<T> extends Iterator<T> {
  length: number,
  limit(lim: number),
}

export interface SerializedNetwork {
  layers: number[];
  weights: Matrix[];
  biases: Matrix[];
  options: NetworkOptionsRequired;
}