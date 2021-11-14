import { mnist } from "../mnist"
import { Network } from "../domain/Network";
import { Matrix } from "../domain/types";
import { saveNetwork, loadNetwork, saveAny } from "./utilities";

function average(inputs: number[]) : number {
  const sum = inputs.reduce((partialSum, el) => partialSum + el, 0);
  return sum / inputs.length;
}

function maxIdx(inputs: Matrix): number {
  let max = 0;
  let idx = -1;
  for (let i = 0; i < inputs.length; i++) {
    const cell = inputs[i][0];
    if (cell > max) {
      idx = i;
      max = cell;
    }
  }
  return idx;
}

async function learnMnist() {
  // const layers = [784, 200, 80, 10];
  // const network = new Network(layers, {learningRate: 0.01, batchSize: 1, progress: true}); 
  const network = await loadNetwork("MNIST");
  network.updateOptions({batchSize: 512 })
  const epochs = 2;
  for (let i = 0; i < epochs; i++) {
    const { training } = await mnist();
    network.learn(training.images, training.labels);
    console.log(`Learning epoch ${i} over`);
    network.updateOptions({ batchSize: network.options.batchSize * 2 }); // increase batch size each epoch
  } 
  saveNetwork(network, "MNIST-2");

  const { test } = await mnist();

  let correct = 0;
  let i = test.images.next();
  let e = test.labels.next();

  while (!i.done && !e.done) {
    const output = network.feed_forward(i.value);
    const prediction = maxIdx(output);
    const expected = maxIdx(e.value);
    if (prediction === expected) correct++;
    i = test.images.next();
    e = test.labels.next();
  }
  console.log(correct/test.images.length);

}

// learnMnist().catch(console.error);

async function mnistToJson() {
  const { test } = await mnist();
  const images = [];
  const labels = [];
  test.images.limit(100);
  test.labels.limit(100);
  let i = test.images.next();
  let l = test.labels.next();

  while (!i.done && !l.done) {
    images.push(i.value);
    labels.push(l.value);
    i = test.images.next();
    l = test.labels.next();
  }
  saveAny("MNIST_TEST", {images, labels});
}
mnistToJson().catch(console.error);
