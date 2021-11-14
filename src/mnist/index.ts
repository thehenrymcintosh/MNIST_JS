import { createMatrix } from '../domain/utilities';
import { IdxTensor, loadBits } from 'idx-data';
import path from 'path';
import { FiniteIterator, Matrix } from '../domain/types';

export function zip(images: Image[], labels: Label[] ): [Image, Label][] {
  if (images.length !== labels.length) throw new Error("Arrays must be same length to zip!");
  const zipped : [Image, Label][] = [];
  for (let i = 0; i < images.length; i++) {
    zipped.push([images[i], labels[i]]);
  }
  return zipped
}


type Image = Matrix;
type Label = Matrix;


type MnistSet = {
  images: FiniteIterator<Image>,
  labels: FiniteIterator<Label>,
}

type MnistDB = {
  training: MnistSet, 
  test: MnistSet
};

function toChunks<T>(data: Iterable<T>, size: number) {
  const chunks: T[][] = [];
  let i = 0;
  for (const el of data) {
    if (i%size === 0) chunks.push([]);
    chunks[chunks.length - 1].push(el);
    i++;
  }
  return chunks;
}

class ImageIterator implements FiniteIterator<Image> {
  private tensor: IdxTensor;
  length: number;
  private position = 0;
  private chunkSize = 784;

  constructor(tensor: IdxTensor, limit?: number) {
    this.tensor = tensor;
    this.length = this.tensor.shape[0]
  }

  limit(lim: number) {
    if (lim < this.length) this.length = lim;
  }

  next() : IteratorResult<Image, null> {
    const pos = this.position;
    if (pos >= this.length) return { done: true, value: null }
    const startPosition = pos * this.chunkSize;
    const endPosition = (pos + 1) * this.chunkSize;
    this.position++;
    const image = Array.from(this.tensor.data.slice(startPosition, endPosition));
    return {
      done: false,
      value: createMatrix(1, 784, (x,y) => image[y])
    }
  }
}

class LabelIterator implements FiniteIterator<Label> {
  private tensor: IdxTensor;
  length: number;
  private position = 0;

  constructor(tensor: IdxTensor) {
    this.tensor = tensor;
    this.length = this.tensor.shape[0]
  }

  limit(lim: number) {
    if (lim < this.length) this.length = lim;
  }

  next() : IteratorResult<Label, null> {
    const pos = this.position;
    if (pos >= this.length) return { done: true, value: null }
    const label = this.tensor.data[pos];
    this.position ++;
    return {
      done: false,
      value: createMatrix(1, 10, (x,y) => y === label ? 1 : 0),
    }
  }
}

async function openImages(fname: string) : Promise<FiniteIterator<Image>> {
  const tensor = await loadBits(path.resolve(__dirname, fname));
  return new ImageIterator(tensor);
}

async function openLabels(fname: string) : Promise<FiniteIterator<Label>> {
  const tensor = await loadBits(path.resolve(__dirname, fname));
  return new LabelIterator(tensor);
}


export async function mnist() : Promise<MnistDB> {
  const [
    training_images,
    training_labels,
    test_images,
    test_labels,
  ] = await Promise.all([
    openImages('train-images-idx3-ubyte'),
    openLabels('train-labels-idx1-ubyte'),
    openImages('t10k-images-idx3-ubyte'),
    openLabels('t10k-labels-idx1-ubyte'),
  ]);

  return {
    training: { images: training_images, labels: training_labels },
    test: { images: test_images, labels: test_labels },
  }
}