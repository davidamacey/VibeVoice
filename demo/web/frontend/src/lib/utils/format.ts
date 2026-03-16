export const pad2 = (value: number): string => value.toString().padStart(2, '0');
export const pad3 = (value: number): string => value.toString().padStart(3, '0');

export function formatLocalTimestamp(): string {
	const d = new Date();
	return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}.${pad3(d.getMilliseconds())}`;
}

export function formatSeconds(raw: number | string): string {
	const value = Number(raw);
	return Number.isFinite(value) ? value.toFixed(2) : '0.00';
}

export function parseTimestamp(value: string | undefined): Date {
	if (!value) return new Date();
	if (/\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}/.test(value)) {
		return new Date(value.replace(' ', 'T'));
	}
	return new Date(value);
}

export function formatTs(t: number | undefined): string {
	if (typeof t !== 'number') return '?';
	const m = Math.floor(t / 60);
	const s = (t % 60).toFixed(2).padStart(5, '0');
	return `${m}:${s}`;
}
