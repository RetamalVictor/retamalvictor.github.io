import { Navigation } from '../utils/navigation.js';
import { seo } from '../utils/seo.js';

interface Topic {
    title: string;
    icon: string;
    tags: string[];
}

const TOPICS: Topic[] = [
    {
        title: 'Mathematics for ML',
        icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4.745 3A23.933 23.933 0 003 12c0 3.183.62 6.22 1.745 9M19.5 3c.967 2.78 1.5 5.817 1.5 9s-.533 6.22-1.5 9M8.25 8.885l1.444-.89a.75.75 0 011.11.649v6.712a.75.75 0 01-1.11.649l-1.444-.89m4.5-6.13h3m-3 3h3" /></svg>',
        tags: ['Linear Algebra', 'Calculus', 'Probability', 'Optimization'],
    },
    {
        title: 'Machine Learning',
        icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5" /></svg>',
        tags: ['Supervised', 'Unsupervised', 'Model Selection', 'Feature Engineering'],
    },
    {
        title: 'Deep Learning',
        icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25z" /></svg>',
        tags: ['PyTorch', 'CNNs', 'Transformers', 'Training Pipelines'],
    },
    {
        title: 'Reinforcement Learning',
        icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15.59 14.37a6 6 0 01-5.84 7.38v-4.8m5.84-2.58a14.98 14.98 0 006.16-12.12A14.98 14.98 0 009.631 8.41m5.96 5.96a14.926 14.926 0 01-5.841 2.58m-.119-8.54a6 6 0 00-7.381 5.84h4.8m2.581-5.84a14.927 14.927 0 00-2.58 5.84m2.699 2.7c-.103.021-.207.041-.311.06a15.09 15.09 0 01-2.448-2.448 14.9 14.9 0 01.06-.312m-2.24 2.39a4.493 4.493 0 00-1.757 4.306 4.493 4.493 0 004.306-1.758M16.5 9a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0z" /></svg>',
        tags: ['MDPs', 'Policy Gradient', 'Multi-Agent RL', 'Simulation'],
    },
    {
        title: 'Computer Vision',
        icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>',
        tags: ['Detection', 'Segmentation', 'Visual Servoing', 'Depth'],
    },
    {
        title: 'ML Engineering & Deployment',
        icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" /></svg>',
        tags: ['Inference', 'ONNX', 'Docker', 'ML Pipelines'],
    },
    {
        title: 'Software Engineering',
        icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M2.25 12.75V12A2.25 2.25 0 014.5 9.75h15A2.25 2.25 0 0121.75 12v.75m-8.69-6.44l-2.12-2.12a1.5 1.5 0 00-1.061-.44H4.5A2.25 2.25 0 002.25 6v12a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9a2.25 2.25 0 00-2.25-2.25h-5.379a1.5 1.5 0 01-1.06-.44z" /></svg>',
        tags: ['Design Patterns', 'Systems Design', 'Architecture', 'Testing'],
    },
    {
        title: 'Thesis & Project Guidance',
        icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" /></svg>',
        tags: ['BSc/MSc Thesis', 'Scoping', 'Experiments', 'Writing'],
    },
];

// Decorative neural network SVG
const NEURAL_NET_SVG = `<svg viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="40" cy="40" r="5" stroke="currentColor" stroke-width="1.5" fill="none"/>
    <circle cx="40" cy="100" r="5" stroke="currentColor" stroke-width="1.5" fill="none"/>
    <circle cx="40" cy="160" r="5" stroke="currentColor" stroke-width="1.5" fill="none"/>
    <circle cx="100" cy="60" r="5" stroke="currentColor" stroke-width="1.5" fill="none"/>
    <circle cx="100" cy="100" r="5" stroke="currentColor" stroke-width="1.5" fill="none"/>
    <circle cx="100" cy="140" r="5" stroke="currentColor" stroke-width="1.5" fill="none"/>
    <circle cx="160" cy="80" r="5" stroke="#fbbf24" stroke-width="1.5" fill="none"/>
    <circle cx="160" cy="120" r="5" stroke="#fbbf24" stroke-width="1.5" fill="none"/>
    <line x1="45" y1="40" x2="95" y2="60" stroke="currentColor" stroke-width="0.8"/>
    <line x1="45" y1="40" x2="95" y2="100" stroke="currentColor" stroke-width="0.8"/>
    <line x1="45" y1="100" x2="95" y2="60" stroke="currentColor" stroke-width="0.8"/>
    <line x1="45" y1="100" x2="95" y2="100" stroke="currentColor" stroke-width="0.8"/>
    <line x1="45" y1="100" x2="95" y2="140" stroke="currentColor" stroke-width="0.8"/>
    <line x1="45" y1="160" x2="95" y2="100" stroke="currentColor" stroke-width="0.8"/>
    <line x1="45" y1="160" x2="95" y2="140" stroke="currentColor" stroke-width="0.8"/>
    <line x1="105" y1="60" x2="155" y2="80" stroke="currentColor" stroke-width="0.8"/>
    <line x1="105" y1="60" x2="155" y2="120" stroke="currentColor" stroke-width="0.8"/>
    <line x1="105" y1="100" x2="155" y2="80" stroke="currentColor" stroke-width="0.8"/>
    <line x1="105" y1="100" x2="155" y2="120" stroke="currentColor" stroke-width="0.8"/>
    <line x1="105" y1="140" x2="155" y2="80" stroke="currentColor" stroke-width="0.8"/>
    <line x1="105" y1="140" x2="155" y2="120" stroke="currentColor" stroke-width="0.8"/>
</svg>`;

export class TutoringPage {
    private container: HTMLElement;

    constructor(container: HTMLElement) {
        this.container = container;
    }

    public async render(): Promise<void> {
        seo.tutoring();
        this.renderPage();
        this.setupEventListeners();
        this.setupScrollAnimations();
    }

    private renderPage(): void {
        document.title = 'Tutoring - Victor Retamal';

        this.container.innerHTML = `
            <div class="min-h-screen bg-dark-bg">
                <!-- Header -->
                <header class="bg-dark-surface/80 backdrop-blur-md border-b border-dark-border sticky top-0 z-40">
                    <div class="max-w-4xl mx-auto px-6 py-5">
                        <div class="flex justify-between items-center">
                            <div>
                                <h1 class="text-xl md:text-2xl font-bold text-white tracking-tight">Private Tutoring</h1>
                                <p class="text-gray-400 text-sm">One-on-one ML, math, and software engineering sessions.</p>
                            </div>
                            <div class="flex items-center gap-3 flex-shrink-0">
                                <a href="/tutoring/es" id="lang-toggle" class="text-xs font-medium px-2.5 py-1 rounded-md border border-accent-cyan/30 text-accent-cyan hover:bg-accent-cyan/10 transition-all duration-300">ES</a>
                                <button id="back-btn" class="text-gray-400 hover:text-accent-cyan transition-all duration-300 flex items-center gap-2 text-sm">
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
                                    </svg>
                                    Home
                                </button>
                            </div>
                        </div>
                    </div>
                </header>

                <main class="max-w-4xl mx-auto px-6 pt-12 pb-12 space-y-10">

                    <!-- Hero -->
                    <section class="relative tut-animate">
                        <div class="absolute top-0 right-0 w-72 h-72 opacity-[0.05] -z-10 hidden md:block text-accent-cyan">
                            ${NEURAL_NET_SVG}
                        </div>

                        <div class="text-center max-w-2xl mx-auto relative">
                            <div class="absolute -inset-16 bg-accent-cyan/[0.03] rounded-full blur-3xl -z-10"></div>
                            <h2 class="text-3xl md:text-5xl font-bold text-white mb-6 leading-tight tracking-tight">
                                Learn from someone who<br><span class="text-accent-cyan">builds it</span> for a living
                            </h2>
                            <p class="text-gray-300 text-lg leading-relaxed mb-10 max-w-xl mx-auto">
                                8+ years in tech and ML, 3 of those teaching university courses.
                                I've published research, deployed models on real robots, and guided students
                                through exams, theses, and career switches. If you want to learn from someone
                                who's done both the building and the teaching, let's talk.
                            </p>

                            <!-- Credentials -->
                            <div class="flex flex-wrap justify-center gap-2.5 mb-10">
                                <span class="px-3.5 py-1.5 text-xs font-medium rounded-full border border-accent-cyan/20 text-accent-cyan/90 bg-accent-cyan/[0.05] backdrop-blur-sm">8+ Years in Tech & ML</span>
                                <span class="px-3.5 py-1.5 text-xs font-medium rounded-full border border-amber-400/20 text-amber-400/90 bg-amber-400/[0.05] backdrop-blur-sm">3 Years University Teaching</span>
                                <span class="px-3.5 py-1.5 text-xs font-medium rounded-full border border-accent-cyan/20 text-accent-cyan/90 bg-accent-cyan/[0.05] backdrop-blur-sm">MSc in AI</span>
                                <span class="px-3.5 py-1.5 text-xs font-medium rounded-full border border-accent-cyan/20 text-accent-cyan/90 bg-accent-cyan/[0.05] backdrop-blur-sm">Published Researcher</span>
                            </div>

                            <a href="#contact" class="btn-primary inline-flex items-center gap-2.5 text-base px-8 py-3.5 rounded-xl shadow-lg shadow-accent-cyan/20 hover:shadow-accent-cyan/30 transition-all duration-300">
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21.75 6.75v10.5a2.25 2.25 0 01-2.25 2.25h-15a2.25 2.25 0 01-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25m19.5 0v.243a2.25 2.25 0 01-1.07 1.916l-7.5 4.615a2.25 2.25 0 01-2.36 0L3.32 8.91a2.25 2.25 0 01-1.07-1.916V6.75" />
                                </svg>
                                Get in touch
                            </a>
                        </div>
                    </section>

                    <!-- Topics -->
                    <section class="tut-animate relative">
                        <!-- Subtle background glow -->
                        <div class="absolute -inset-8 bg-accent-purple/[0.025] rounded-3xl blur-2xl -z-10"></div>

                        <div class="bg-dark-surface/50 backdrop-blur-sm border border-dark-border rounded-2xl p-8 md:p-10">
                            <h2 class="text-2xl font-bold text-white mb-8 tracking-tight">
                                What I <span class="text-accent-cyan">teach</span>
                            </h2>
                            <!-- Column headers -->
                            <div class="grid md:grid-cols-2 gap-x-12 mb-5">
                                <h3 class="text-xs font-semibold uppercase tracking-[0.15em] text-accent-cyan/80 flex items-center gap-2.5">
                                    <span class="w-8 h-px bg-accent-cyan/30"></span>
                                    Foundations
                                </h3>
                                <h3 class="text-xs font-semibold uppercase tracking-[0.15em] text-accent-purple/80 flex items-center gap-2.5 mt-6 md:mt-0">
                                    <span class="w-8 h-px bg-accent-purple/30"></span>
                                    Applied
                                </h3>
                            </div>
                            <!-- Topic rows: shared grid so rows align across columns -->
                            <div class="grid md:grid-cols-2 gap-x-12">
                                ${[0, 1, 2, 3].map(i => `
                                    ${this.renderTopicItem(TOPICS[i])}
                                    ${this.renderTopicItem(TOPICS[i + 4])}
                                `).join('')}
                            </div>
                        </div>
                    </section>

                    <!-- Why me — full-width strip via negative margins -->
                    <section class="tut-animate -mx-6 px-6 py-10 relative">
                        <!-- Full-width background -->
                        <div class="absolute inset-0 -z-10 overflow-hidden" style="left: calc(-50vw + 50%); right: calc(-50vw + 50%)">
                            <div class="absolute inset-0" style="background: radial-gradient(ellipse 80% 60% at 30% 50%, rgba(168,85,247,0.04) 0%, transparent 70%)"></div>
                            <div class="absolute inset-0" style="background: radial-gradient(ellipse 60% 50% at 70% 50%, rgba(0,212,255,0.03) 0%, transparent 70%)"></div>
                        </div>

                        <div>
                            <h2 class="text-2xl font-bold text-white mb-8 tracking-tight">
                                Why learn <span class="text-accent-purple">with me</span>
                            </h2>

                            <div class="grid md:grid-cols-[1.2fr,1fr] gap-8 md:gap-12 items-start">
                                <!-- Featured point (left) -->
                                <div class="relative group">
                                    <div class="absolute -inset-4 bg-accent-cyan/[0.04] rounded-2xl blur-xl -z-10 group-hover:bg-accent-cyan/[0.07] transition-all duration-500"></div>
                                    <div class="bg-dark-surface/60 backdrop-blur-sm border border-accent-cyan/15 rounded-2xl p-7 group-hover:border-accent-cyan/30 transition-all duration-300">
                                        <h3 class="text-xl font-bold text-white mb-2 tracking-tight">I build and I teach.</h3>
                                        <p class="text-gray-400 text-sm leading-relaxed mb-4">
                                            The interactive demos on this site run ML models in your browser.
                                            I don't just know the theory, I ship it to production.
                                        </p>
                                        <a href="/" class="text-accent-cyan hover:text-white transition-all duration-300 text-sm font-medium inline-flex items-center gap-1.5 group/link">
                                            See my demos
                                            <svg class="w-4 h-4 group-hover/link:translate-x-0.5 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                                            </svg>
                                        </a>
                                    </div>
                                </div>

                                <!-- Stacked points (right) -->
                                <div class="space-y-5 md:pt-1">
                                    <div class="flex items-start gap-3.5">
                                        <span class="text-accent-cyan font-bold text-lg leading-none mt-0.5 flex-shrink-0 w-7">3y</span>
                                        <div>
                                            <span class="text-white font-medium text-sm">University Teaching</span>
                                            <p class="text-gray-500 text-xs mt-0.5 leading-relaxed">TA at VU Amsterdam. Wrote exams, designed assignments, graded two BSc/MSc courses.</p>
                                        </div>
                                    </div>
                                    <div class="w-full h-px bg-gradient-to-r from-dark-border via-dark-border/50 to-transparent"></div>
                                    <div class="flex items-start gap-3.5">
                                        <span class="text-accent-cyan font-bold text-lg leading-none mt-0.5 flex-shrink-0 w-7">8+</span>
                                        <div>
                                            <span class="text-white font-medium text-sm">Years in Tech & ML</span>
                                            <p class="text-gray-500 text-xs mt-0.5 leading-relaxed">From research labs to industry. Published in MARL, deployed ML on real robots.</p>
                                        </div>
                                    </div>
                                    <div class="w-full h-px bg-gradient-to-r from-dark-border via-dark-border/50 to-transparent"></div>
                                    <div class="flex items-start gap-3.5">
                                        <span class="text-accent-cyan font-bold text-lg leading-none mt-0.5 flex-shrink-0 w-7">1:1</span>
                                        <div>
                                            <span class="text-white font-medium text-sm">Tailored to You</span>
                                            <p class="text-gray-500 text-xs mt-0.5 leading-relaxed">No generic curriculum. We build a plan around your goals and current level.</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </section>

                    <!-- How it works + Contact -->
                    <section id="contact" class="tut-animate relative rounded-2xl p-[1px]" style="background: linear-gradient(135deg, rgba(0,212,255,0.4), rgba(168,85,247,0.3), rgba(251,191,36,0.2), rgba(0,212,255,0.4))">
                        <div class="bg-dark-surface rounded-2xl p-8 md:p-10 relative overflow-hidden">
                            <div class="absolute top-0 right-0 w-72 h-72 bg-accent-cyan/[0.04] rounded-full blur-3xl"></div>
                            <div class="absolute bottom-0 left-0 w-56 h-56 bg-accent-purple/[0.03] rounded-full blur-3xl"></div>

                            <div class="grid md:grid-cols-[1fr,auto,1fr] gap-8 md:gap-0 items-center relative">
                                <!-- Left: Steps -->
                                <div class="md:pr-10">
                                    <h2 class="text-2xl font-bold text-white mb-6 tracking-tight">
                                        How to <span class="text-accent-cyan">get started</span>
                                    </h2>
                                    <div class="space-y-5">
                                        ${this.renderStep('1', 'Reach out', 'Tell me what you want to learn.')}
                                        ${this.renderStep('2', 'Free intro call', 'We check if it\'s a good fit. No cost.')}
                                        ${this.renderStep('3', 'Start learning', '1-on-1 video sessions, at your pace.')}
                                    </div>
                                </div>

                                <!-- Divider -->
                                <div class="hidden md:block w-px h-48 bg-gradient-to-b from-transparent via-dark-border to-transparent"></div>
                                <div class="md:hidden h-px w-full bg-gradient-to-r from-transparent via-dark-border to-transparent"></div>

                                <!-- Right: CTA -->
                                <div class="md:pl-10 text-center md:text-left">
                                    <h3 class="text-2xl font-bold text-white mb-3 tracking-tight">
                                        Ready to <span class="text-accent-cyan">start</span>?
                                    </h3>
                                    <p class="text-gray-400 text-sm mb-6 leading-relaxed">
                                        The first intro call is on me. Tell me a bit about yourself and what you'd like to work on.
                                    </p>
                                    <div class="flex flex-col gap-3 items-center md:items-start">
                                        <a href="mailto:retamal1.victor@gmail.com?subject=Tutoring%20Inquiry" class="btn-primary inline-flex items-center gap-2.5 text-base px-8 py-3.5 rounded-xl shadow-lg shadow-accent-cyan/20 hover:shadow-accent-cyan/30 transition-all duration-300">
                                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21.75 6.75v10.5a2.25 2.25 0 01-2.25 2.25h-15a2.25 2.25 0 01-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25m19.5 0v.243a2.25 2.25 0 01-1.07 1.916l-7.5 4.615a2.25 2.25 0 01-2.36 0L3.32 8.91a2.25 2.25 0 01-1.07-1.916V6.75" />
                                            </svg>
                                            Email me
                                        </a>
                                        <!-- Superprof booking link placeholder -->
                                        <a id="superprof-link" href="#" class="btn-secondary inline-flex items-center gap-2.5 text-base px-8 py-3.5 rounded-xl transition-all duration-300" style="display: none;">
                                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 012.25-2.25h13.5A2.25 2.25 0 0121 7.5v11.25m-18 0A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75m-18 0v-7.5A2.25 2.25 0 015.25 9h13.5A2.25 2.25 0 0121 11.25v7.5" />
                                            </svg>
                                            Book a session
                                        </a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </section>

                </main>
            </div>
        `;
    }

    private renderTopicItem(topic: Topic): string {
        return `
            <div class="group flex items-center gap-3.5 py-3 px-3 -mx-3 rounded-xl transition-all duration-300 hover:bg-white/[0.02] min-h-[4rem]">
                <div class="w-10 h-10 rounded-lg bg-accent-cyan/[0.07] border border-accent-cyan/15 flex items-center justify-center flex-shrink-0 group-hover:bg-accent-cyan/[0.14] group-hover:border-accent-cyan/30 transition-all duration-300">
                    <div class="text-accent-cyan/70 group-hover:text-accent-cyan transition-colors duration-300">${topic.icon}</div>
                </div>
                <div>
                    <span class="text-white font-medium text-sm group-hover:text-accent-cyan transition-colors duration-300">${topic.title}</span>
                    <p class="text-gray-500 group-hover:text-gray-400 text-sm mt-0.5 transition-colors duration-300 leading-snug">${topic.tags.join(' · ')}</p>
                </div>
            </div>
        `;
    }

    private renderStep(number: string, title: string, description: string): string {
        return `
            <div class="flex items-start gap-4 group">
                <div class="w-9 h-9 rounded-full bg-accent-cyan/10 border border-accent-cyan/30 flex items-center justify-center flex-shrink-0 group-hover:bg-accent-cyan/20 group-hover:border-accent-cyan/50 transition-all duration-300">
                    <span class="text-accent-cyan font-semibold text-sm">${number}</span>
                </div>
                <div>
                    <h3 class="text-white font-semibold text-sm mb-0.5">${title}</h3>
                    <p class="text-gray-500 text-sm">${description}</p>
                </div>
            </div>
        `;
    }

    private setupEventListeners(): void {
        const backBtn = document.getElementById('back-btn');
        backBtn?.addEventListener('click', () => {
            Navigation.toHome();
        });

        const langToggle = document.getElementById('lang-toggle');
        langToggle?.addEventListener('click', (e) => {
            e.preventDefault();
            Navigation.to('/tutoring/es');
        });

        this.container.querySelectorAll('a[href="#contact"]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                document.getElementById('contact')?.scrollIntoView({ behavior: 'smooth' });
            });
        });
    }

    private setupScrollAnimations(): void {
        const sections = this.container.querySelectorAll('.tut-animate');
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    (entry.target as HTMLElement).style.opacity = '1';
                    (entry.target as HTMLElement).style.transform = 'translateY(0)';
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });

        sections.forEach(section => {
            (section as HTMLElement).style.opacity = '0';
            (section as HTMLElement).style.transform = 'translateY(20px)';
            (section as HTMLElement).style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
            observer.observe(section);
        });
    }

    public destroy(): void {
        // Cleanup if needed
    }
}
