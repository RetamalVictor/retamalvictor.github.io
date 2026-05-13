/**
 * YouTubeEmbed - Creates a YouTube embed by injecting an iframe via JS.
 * Bypasses DOMPurify which blocks iframes in sanitized HTML.
 */

export interface YouTubeEmbedConfig {
    containerId: string;
    videoId?: string;
    caption?: string;
}

export class YouTubeEmbed {
    private container: HTMLElement;
    private wrapper: HTMLElement | null = null;

    constructor(config: YouTubeEmbedConfig) {
        const container = document.getElementById(config.containerId);
        if (!container) {
            throw new Error(`Container #${config.containerId} not found`);
        }
        this.container = container;

        const videoId = config.videoId ?? 'XxzP5Dqwdmc';
        const caption = config.caption ?? '';

        this.render(videoId, caption);
    }

    private render(videoId: string, caption: string): void {
        // Create wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'youtube-embed-container';

        // Create iframe
        const iframe = document.createElement('iframe');
        iframe.src = `https://www.youtube-nocookie.com/embed/${videoId}`;
        iframe.setAttribute('allow', 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture');
        iframe.setAttribute('allowfullscreen', '');
        iframe.setAttribute('loading', 'lazy');
        iframe.style.position = 'absolute';
        iframe.style.top = '0';
        iframe.style.left = '0';
        iframe.style.width = '100%';
        iframe.style.height = '100%';
        iframe.style.border = 'none';

        wrapper.appendChild(iframe);
        this.container.appendChild(wrapper);

        // Add caption if provided
        if (caption) {
            const captionEl = document.createElement('p');
            captionEl.className = 'embed-caption';
            captionEl.textContent = caption;
            this.container.appendChild(captionEl);
        }

        this.wrapper = wrapper;
    }

    destroy(): void {
        if (this.wrapper) {
            this.wrapper.remove();
            this.wrapper = null;
        }
        // Also remove any caption
        const caption = this.container.querySelector('.embed-caption');
        if (caption) {
            caption.remove();
        }
    }
}
