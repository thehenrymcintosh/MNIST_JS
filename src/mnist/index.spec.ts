import {expect} from "chai";
import mocha from "mocha";
const { describe, it } = mocha;
import { mnist } from "./index";

describe("MNIST", () => {
  it("Loads MNIST images correctly", async () => {
    const { training, test } = await mnist(); 
    expect(training.images).lengthOf(60000);
    expect(training.labels).lengthOf(60000);
    expect(test.images).lengthOf(10000);
    expect(test.labels).lengthOf(10000);
    for (let i = 0; i < 10000; i++) {
      expect(test.images.next().done, `Only had length ${i}`).to.be.false
    }
    expect(test.images.next().done, `Had more than expected`).to.be.true

  }).timeout(10000);
})