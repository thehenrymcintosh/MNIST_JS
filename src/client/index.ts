import { Matrix, SerializedNetwork } from "../domain/types";
import { Network } from "../domain/Network";
import MNIST from "../models/MNIST.json";
import MNIST2 from "../models/MNIST-2.json";
import MNIST_TEST_DATA from "../models/MNIST_TEST.json";
import { createMatrix } from "../domain/utilities";

class DrawableCanvas {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  mousePosition = {x: 0, y: 0};
  drawCallbacks: VoidFunction[] = [];
  writeCallbacks: VoidFunction[] = [];

  constructor() {
    this.canvas = document.createElement('canvas');
    document.body.appendChild(this.canvas);
    var ctx = this.canvas.getContext('2d');
    if (ctx === null) throw new Error("can't get canvas context");
    this.ctx = ctx;
    this.ctx.canvas.width = 560;
    this.ctx.canvas.height = 560;
    this.clear();

    this.mouseDraw = this.mouseDraw.bind(this);
    this.touchDraw = this.touchDraw.bind(this);
    this.setPosition = this.setPosition.bind(this);
    this.setTouchPosition = this.setTouchPosition.bind(this);
    this.clear = this.clear.bind(this);

    document.addEventListener('mousemove', this.mouseDraw);
    document.addEventListener('touchmove', this.touchDraw);
    document.addEventListener('mousedown', this.setPosition);
    document.addEventListener('touchstart', this.setTouchPosition);
    document.addEventListener('mouseup', () => {
      this.write(this.pixellate());
      this.drawCallbacks.forEach(cb => cb());
    });
    document.addEventListener('touchEnd', () => {
      this.write(this.pixellate());
      this.drawCallbacks.forEach(cb => cb());
    });
    document.addEventListener('mouseenter', this.setPosition);
  }

  private windowToLocalMousePos(evt: {clientX: number, clientY: number}) {
    var rect = this.canvas.getBoundingClientRect(), // abs. size of element
        scaleX = this.canvas.width / rect.width,    // relationship bitmap vs. element for X
        scaleY = this.canvas.height / rect.height;  // relationship bitmap vs. element for Y
    return {
      x: (evt.clientX - rect.left) * scaleX,   // scale mouse coordinates after they have
      y: (evt.clientY - rect.top) * scaleY     // been adjusted to be relative to element
    }
  }

  private setPosition(e: MouseEvent) {
    this.mousePosition = this.windowToLocalMousePos(e);
  }
  
  
  private setTouchPosition(e: TouchEvent) {
    this.mousePosition = this.windowToLocalMousePos(e.touches[0]);
  }
    

  private mouseDraw(e: MouseEvent) {
    const { ctx } = this;
    // mouse left button must be pressed
    if (e.buttons !== 1) return;
  
    ctx.beginPath(); // begin
  
    ctx.lineWidth = 50;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#ffffff';
  
    ctx.moveTo(this.mousePosition.x, this.mousePosition.y); // from
    this.setPosition(e);
    ctx.lineTo(this.mousePosition.x, this.mousePosition.y); // to
  
    ctx.stroke(); // draw it!
    this.drawCallbacks.forEach(cb => cb());
  }

  private touchDraw(e: TouchEvent) {
    const { ctx } = this;
  
    ctx.beginPath(); // begin
  
    ctx.lineWidth = 50;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#ffffff';
  
    ctx.moveTo(this.mousePosition.x, this.mousePosition.y); // from
    this.setTouchPosition(e);
    ctx.lineTo(this.mousePosition.x, this.mousePosition.y); // to
  
    ctx.stroke(); // draw it!
    this.drawCallbacks.forEach(cb => cb());
  }

  write(pixels: number[][]) {
    this.clear();
    const widthInPixels = pixels.length; // square canvas;
    const pxSize = this.ctx.canvas.width / widthInPixels;
    for (let y = 0; y < widthInPixels; y++) {
      for (let x = 0; x < widthInPixels; x++) {
        this.ctx.fillStyle = `rgb(${pixels[y][x]},${pixels[y][x]},${pixels[y][x]})`;
        this.ctx.fillRect(x*pxSize, y*pxSize, pxSize, pxSize);
      } 
    }
    this.writeCallbacks.forEach(cb => cb());
  }

  clear() {
    this.ctx.fillStyle = "#000000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  pixellate() {
    const widthInPixels = 28; // square canvas;
    const pxSize = this.ctx.canvas.width / widthInPixels;
    const pixels: number[][] = [];
    for (let y = 0; y < widthInPixels; y++) {
      const row: number[] = [];
      for (let x = 0; x < widthInPixels; x++) {
        row.push( this.brightness(x*pxSize, y*pxSize, pxSize, pxSize) );
      } 
      pixels.push(row);
    }
    return pixels;
  }

  private brightness(x: number, y: number, w: number, h: number) {
    const imageData = this.ctx.getImageData(x, y, w, h);
    const data = imageData.data;
    let colorSum = 0;
    let r, g, b, avg;
    for(let x=0, len=data.length; x<len; x+=4) {
      r = data[x];
      g = data[x+1];
      b = data[x+2];
      avg = Math.floor((r+g+b) / 3);
      colorSum += avg;
    }

    const brightness = Math.floor(colorSum / (w * h));
    return brightness;
  }

  onDraw(callback: VoidFunction) {
    this.drawCallbacks.push(callback);
  }
  onWrite(callback: VoidFunction) {
    this.writeCallbacks.push(callback);
  }
}


class RenderCanvas {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
 
  constructor() {
    this.canvas = document.createElement('canvas');
    document.body.appendChild(this.canvas);
    var ctx = this.canvas.getContext('2d');
    if (ctx === null) throw new Error("can't get canvas context");
    this.ctx = ctx;
    this.ctx.canvas.width = 28;
    this.ctx.canvas.height = 28;
    this.clear();
  }

  write(pixels: number[][]) {
    this.clear();
    const widthInPixels = this.ctx.canvas.width; // square canvas;
    const pxSize = this.ctx.canvas.width / widthInPixels;
    for (let y = 0; y < widthInPixels; y++) {
      for (let x = 0; x < widthInPixels; x++) {
        this.ctx.fillStyle = `rgb(${pixels[y][x]},${pixels[y][x]},${pixels[y][x]})`;
        this.ctx.fillRect(x*pxSize, y*pxSize, pxSize, pxSize);
      } 
    }
  }

  clear() {
    this.ctx.fillStyle = "#000000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }
}

function maxIdx(inputs: Matrix): [number, number] {
  let max = 0;
  let idx = -1;
  for (let i = 0; i < inputs.length; i++) {
    const cell = inputs[i][0];
    if (cell > max) {
      idx = i;
      max = cell;
    }
  }
  return [max, idx];
}

class Predictor {
  private div: HTMLDivElement;
  private div2: HTMLDivElement;
  private model: Network;
  readonly name: string;
  constructor(modelName: string, model: SerializedNetwork) {
    this.div = document.createElement('div');
    this.div2 = document.createElement('div');
    document.body.appendChild(this.div);
    document.body.appendChild(this.div2);
    this.model = new Network([]);
    this.model.load(model);
    this.name = modelName;
  }

  private prepareImage(inputs: Matrix): Matrix {
    // convert from 28x28 to 1x784;
    return createMatrix(1, 784, (_,pos) => {
      const y = Math.floor(pos / 28);
      const x = pos % 28;
      return inputs[y][x];
    } )
  }

  predict(pixels: number[][]) {
    const image = this.prepareImage(pixels);
    const result = this.model.feed_forward(image);
    const [confidence, prediction] = maxIdx(result);
    this.div.textContent = `${this.name} predicts ${prediction} with confidence ${confidence}`;
    this.div2.textContent = `${result.map((arr, i) => `${i}: ${arr[0].toFixed(2)}`).join(", ")}`;
  }
}



class Tester {
  private div: HTMLDivElement;
  private button: HTMLButtonElement;
  constructor(cb: (data: {image: Matrix, label: Matrix}) => void) {
    this.div = document.createElement('div');
    this.button = document.createElement('button');
    this.clear();
    this.button.textContent = "Random number from test data";
    this.button.addEventListener("click", () => {
      cb( this.random() );
    })
    document.body.appendChild(this.div);
    document.body.appendChild(this.button);
  }

  random(): {image: Matrix, label: Matrix} {
    const data = MNIST_TEST_DATA as { images: Matrix[], labels: Matrix[] };
    const idx = Math.floor(Math.random() * data.images.length);
    const columnImage = data.images[idx]; // will be 1x768;
    const image = createMatrix(28, 28, (x,y) => columnImage[y*28 + x][0]);
    const label = data.labels[idx];
    this.div.textContent = `Labelled as: ${maxIdx(label)[1]}`;
    
    return {image, label};
  }

  clear() {
    this.div.textContent = "";
  }


}

function init() {
  const drawable = new DrawableCanvas();
  (window as any).clearCanvas = drawable.clear;
  const preVis = new RenderCanvas();
  const predictors = [
    new Predictor("Model 1", MNIST),
    new Predictor("Model 2", MNIST2),
  ]
  drawable.onDraw(() => preVis.write(drawable.pixellate()));
  drawable.onWrite(() => preVis.write(drawable.pixellate()));
  predictors.forEach(predictor => {
    drawable.onDraw(() => predictor.predict(drawable.pixellate()) );
    drawable.onWrite(() => predictor.predict(drawable.pixellate()) );
  })
  const tester = new Tester(({image}) => drawable.write(image));
  drawable.onDraw(() => tester.clear());

}

window.addEventListener("load", init);