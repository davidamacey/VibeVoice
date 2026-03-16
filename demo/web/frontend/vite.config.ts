import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		proxy: {
			'/config': 'http://localhost:8000',
			'/models': 'http://localhost:8000',
			'/generate': 'http://localhost:8000',
			'/transcribe': 'http://localhost:8000',
			'/stream': {
				target: 'ws://localhost:8000',
				ws: true
			}
		}
	}
});
