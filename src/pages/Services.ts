import { Navigation } from '../utils/navigation.js';
import { seo } from '../utils/seo.js';
import { openMail } from '../utils/contact.js';
import { refreshThemeToggles, themeToggleMarkup } from '../utils/theme.js';

interface Service {
    title: string;
    icon: string;
    description: string;
}

const SERVICES: Service[] = [
    {
        title: 'Sim-to-Real Infrastructure',
        icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>',
        description: 'Simulation environments, domain randomization, distributed training pipelines, and iterative hardware validation to close the sim-to-real gap.',
    },
    {
        title: 'Multi-Agent RL & Robotics',
        icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" /></svg>',
        description: 'Environment design, reward shaping, policy training, evaluation benchmarks, and deployment on real platforms.',
    },
    {
        title: 'Medical Imaging & Surgical AI',
        icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" /></svg>',
        description: 'Segmentation pipelines, 3D patient-specific models, dataset curation. MD background means I understand the clinical workflow, not just the model.',
    },
    {
        title: 'Computer Vision & Edge Deployment',
        icon: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>',
        description: 'Real-time inference pipelines for embedded devices. Model optimization, quantization, and deployment on Jetson or custom hardware.',
    },
];

const PUBLICATIONS = [
    { venue: 'ICUAS 2026', title: 'Speed-Based Trajectory Tracking Control for Fixed-Wing UAV', doi: '' },
    { venue: 'SAUS 2024', title: 'Automatic Segmentation of Cardiac Structures from 2D Echocardiographic Images using Transformers', doi: '10.1109/SAUS61785.2024.10563657' },
    { venue: 'RAL 2023', title: 'From Shadows to Light: A Swarm Robotics Approach with Onboard Control for Seeking Dynamic Sources in Constrained Environments', doi: '10.1109/LRA.2023.3331897' },
    { venue: 'ICRA 2023', title: 'Onboard Controller Design for Nano UAV Swarm in Operator-Guided Collective Behaviors', doi: '10.1109/ICRA48891.2023.10160630' },
];

export class ServicesPage {
    private container: HTMLElement;

    constructor(container: HTMLElement) {
        this.container = container;
    }

    public async render(): Promise<void> {
        seo.services();
        this.renderPage();
        this.setupEventListeners();
        this.setupScrollAnimations();
    }

    private renderPage(): void {
        document.title = 'Services - Victor Retamal';

        this.container.innerHTML = `
            <div class="min-h-screen">
                <!-- Header -->
                <header class="sticky top-0 z-40">
                    <div class="win-bar">
                        <span class="win-title">/services/index.html</span>
                        <span class="win-controls" aria-hidden="true"><i></i><i></i><i></i></span>
                    </div>
                    <div class="bg-dark-surface border-b border-dark-border">
                        <div class="max-w-4xl mx-auto px-4 sm:px-6 py-4">
                            <div class="flex flex-wrap justify-between items-center gap-3">
                                <div>
                                    <h1 class="heading-retro text-2xl">Services</h1>
                                    <p class="text-sm text-gray-500 mt-1 pl-[1.3rem]">ML engineering, robotics, and consulting</p>
                                </div>
                                <div class="flex items-center gap-2">
                                    <button id="back-btn" class="btn-mini">&#8592; Home</button>
                                    ${themeToggleMarkup()}
                                </div>
                            </div>
                        </div>
                    </div>
                </header>

                <main class="max-w-4xl mx-auto px-4 sm:px-6 py-8 space-y-8">

                    <!-- Intro -->
                    <section class="tut-animate win">
                        <div class="win-bar">
                            <span class="win-title">about.txt</span>
                            <span class="hidden sm:flex items-center gap-1.5 ml-3 normal-case tracking-normal font-normal opacity-90">
                                <span class="led"></span> available
                            </span>
                            <span class="win-controls" aria-hidden="true"><i></i><i></i><i></i></span>
                        </div>
                        <div class="win-body">
                            <p class="prompt mb-5"><span>C:\\&gt; type about.txt</span><span class="caret"></span></p>

                            <h2 class="title-3d text-3xl md:text-4xl font-black text-white mb-3 leading-tight">
                                Victor <span class="text-accent-cyan">Retamal</span>
                            </h2>
                            <p class="pill mb-5">Senior Machine Learning Engineer</p>

                            <p class="text-gray-400 text-lg leading-relaxed mb-6 max-w-2xl">
                                ML and robotics engineer with 6+ years building sim-to-real pipelines, multi-agent systems,
                                computer vision, and medical imaging. 4 IEEE publications and a medical degree
                                that gives me a genuine edge in clinical AI.
                            </p>

                            <table class="specs mb-6">
                                <tbody>
                                    <tr><th scope="row">Available</th><td>Contract work &middot; remote or on-site (Spain/EU) &middot; EU citizen</td></tr>
                                    <tr><th scope="row">Simulation</th><td>Gazebo &middot; Isaac Lab &middot; MuJoCo</td></tr>
                                    <tr><th scope="row">Learning</th><td>PyTorch &middot; JAX &middot; CUDA</td></tr>
                                    <tr><th scope="row">Robotics</th><td>ROS/ROS2 &middot; PX4</td></tr>
                                    <tr><th scope="row">Edge</th><td>TensorRT &middot; ONNX &middot; Jetson</td></tr>
                                    <tr><th scope="row">Systems</th><td>C/C++ &middot; Docker &middot; Kubernetes</td></tr>
                                </tbody>
                            </table>

                            <a href="#contact" class="btn-primary inline-flex items-center gap-2.5">
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21.75 6.75v10.5a2.25 2.25 0 01-2.25 2.25h-15a2.25 2.25 0 01-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25m19.5 0v.243a2.25 2.25 0 01-1.07 1.916l-7.5 4.615a2.25 2.25 0 01-2.36 0L3.32 8.91a2.25 2.25 0 01-1.07-1.916V6.75" />
                                </svg>
                                Get in touch
                            </a>
                        </div>
                    </section>

                    <!-- Services -->
                    <section class="tut-animate">
                        <h2 class="heading-retro text-2xl mb-5 pb-3 border-b-2 border-dotted border-dark-border">What I do</h2>
                        <div class="grid md:grid-cols-2 gap-4">
                            ${SERVICES.map((s, i) => this.renderServiceItem(s, i)).join('')}
                        </div>
                    </section>

                    <!-- Publications -->
                    <section class="tut-animate win">
                        <div class="win-bar">
                            <span class="win-title">publications.bib</span>
                            <span class="win-controls" aria-hidden="true"><i></i><i></i><i></i></span>
                        </div>
                        <div class="win-body">
                            <div class="space-y-2">
                                ${PUBLICATIONS.map(p => {
                                    const tag = p.doi
                                        ? `<a href="https://doi.org/${p.doi}" target="_blank" rel="noopener noreferrer" class="block p-3 rounded border border-dark-border hover:border-accent-cyan hover:bg-dark-hover transition-colors group">`
                                        : `<div class="block p-3 rounded border border-dark-border">`;
                                    const close = p.doi ? '</a>' : '</div>';
                                    return `
                                    ${tag}
                                        <div class="flex items-start gap-3">
                                            <span class="pill flex-shrink-0 mt-0.5">${p.venue}</span>
                                            <p class="text-sm text-gray-400 ${p.doi ? 'group-hover:text-white' : ''} transition-colors">${p.title}</p>
                                        </div>
                                    ${close}`;
                                }).join('')}
                            </div>
                        </div>
                    </section>

                    <!-- Contact -->
                    <section id="contact" class="tut-animate win">
                        <div class="win-bar">
                            <span class="win-title">contact.exe</span>
                            <span class="win-controls" aria-hidden="true"><i></i><i></i><i></i></span>
                        </div>
                        <div class="win-body">
                            <div class="text-center max-w-lg mx-auto">
                                <h2 class="text-2xl font-bold text-white mb-3">
                                    Let's <span class="text-accent-cyan">work together</span>
                                </h2>
                                <p class="text-gray-400 text-sm mb-7 leading-relaxed">
                                    Available for contract work, consulting, and research collaborations. Tell me about your project.
                                </p>
                                <div class="flex flex-col sm:flex-row gap-3 justify-center">
                                    <button id="email-me-btn" type="button" class="btn-primary inline-flex items-center justify-center gap-2.5 text-base px-8 py-3.5">
                                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21.75 6.75v10.5a2.25 2.25 0 01-2.25 2.25h-15a2.25 2.25 0 01-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25m19.5 0v.243a2.25 2.25 0 01-1.07 1.916l-7.5 4.615a2.25 2.25 0 01-2.36 0L3.32 8.91a2.25 2.25 0 01-1.07-1.916V6.75" />
                                        </svg>
                                        Email me
                                    </button>
                                    <a href="https://www.linkedin.com/in/victor-retamal/" target="_blank" rel="noopener noreferrer" class="btn-secondary inline-flex items-center justify-center gap-2.5 text-base px-8 py-3.5">
                                        <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                            <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                                        </svg>
                                        LinkedIn
                                    </a>
                                </div>
                            </div>
                        </div>
                    </section>

                </main>
            </div>
        `;

        refreshThemeToggles();
    }

    private renderServiceItem(service: Service, index: number): string {
        return `
            <div class="card group flex flex-col">
                <span class="win-bar">
                    <span class="win-title">0${index + 1}</span>
                </span>
                <div class="p-5 flex-1">
                    <div class="flex items-center gap-3 mb-2">
                        <span class="w-9 h-9 rounded flex items-center justify-center flex-shrink-0 border border-accent-cyan/30 bg-accent-cyan/10 text-accent-cyan">${service.icon}</span>
                        <h3 class="text-white font-bold text-sm group-hover:text-accent-cyan transition-colors">${service.title}</h3>
                    </div>
                    <p class="text-gray-400 text-sm leading-snug">${service.description}</p>
                </div>
            </div>
        `;
    }

    private setupEventListeners(): void {
        const backBtn = document.getElementById('back-btn');
        backBtn?.addEventListener('click', () => {
            Navigation.toHome();
        });

        document.getElementById('email-me-btn')?.addEventListener('click', () => {
            openMail('Project inquiry');
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
