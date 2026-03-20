class PcmAudioProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0];

    // Copy the Float32Array so it can be sent safely to the main thread
    const copy = new Float32Array(channelData.length);
    copy.set(channelData);

    this.port.postMessage(copy);
    return true;
  }
}

registerProcessor("pcm-audio-processor", PcmAudioProcessor);