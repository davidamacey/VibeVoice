import { SAMPLE_RATE } from '$lib/constants';

export function createWavBlob(chunks: ArrayBuffer[], totalSamples: number): Blob | null {
	if (!totalSamples) return null;

	const wavBuffer = new ArrayBuffer(44 + totalSamples * 2);
	const view = new DataView(wavBuffer);

	const writeString = (offset: number, str: string) => {
		for (let i = 0; i < str.length; i++) {
			view.setUint8(offset + i, str.charCodeAt(i));
		}
	};

	writeString(0, 'RIFF');
	view.setUint32(4, 36 + totalSamples * 2, true);
	writeString(8, 'WAVE');
	writeString(12, 'fmt ');
	view.setUint32(16, 16, true);
	view.setUint16(20, 1, true);
	view.setUint16(22, 1, true);
	view.setUint32(24, SAMPLE_RATE, true);
	view.setUint32(28, SAMPLE_RATE * 2, true);
	view.setUint16(32, 2, true);
	view.setUint16(34, 16, true);
	writeString(36, 'data');
	view.setUint32(40, totalSamples * 2, true);

	const pcmData = new Int16Array(wavBuffer, 44, totalSamples);
	let offset = 0;
	chunks.forEach((chunk) => {
		const chunkData = new Int16Array(chunk);
		pcmData.set(chunkData, offset);
		offset += chunkData.length;
	});

	return new Blob([wavBuffer], { type: 'audio/wav' });
}

export function triggerDownload(blob: Blob, filename: string): void {
	const url = URL.createObjectURL(blob);
	const link = document.createElement('a');
	link.href = url;
	link.download = filename;
	document.body.appendChild(link);
	link.click();
	document.body.removeChild(link);
	URL.revokeObjectURL(url);
}
