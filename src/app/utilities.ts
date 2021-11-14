import { promises } from "fs";
import path from "path";
import { Network } from "../domain/Network";

const { writeFile, readFile } = promises;

function tofilePath(fileName: string): string {
  return path.resolve(__dirname, "../models", `${fileName}.json`);
}

export async function saveNetwork(network: Network, fileName: string) {
  return writeFile(tofilePath(fileName), JSON.stringify(network.serialize()), { flag: "wx" }, )
}


export async function loadNetwork(fileName: string) {
  const n = new Network([]);
  const readBuffer = await readFile( tofilePath(fileName) );
  const serialized = JSON.parse(readBuffer.toString());
  n.load(serialized);
  return n;
}

export function saveAny(fileName: string, data: any) {
  return writeFile(tofilePath(fileName), JSON.stringify(data), { flag: "wx" }, )
}
