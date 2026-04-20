import { Navigation } from '../utils/navigation.js';
import { seo } from '../utils/seo.js';

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
    { venue: 'SAUS 2024', title: 'Automatic Segmentation of Cardiac Structures from 2-D Echocardiographic Images using Transformers', doi: '10.1109/SAUS61785.2024.10563657' },
    { venue: 'RAL 2023', title: 'From Shadows to Light: A Swarm-Robotics Approach with On-board Control for Seeking Dynamic Sources', doi: '10.1109/LRA.2023.3331897' },
    { venue: 'ICRA 2023', title: 'On-board Controller Design for Nano UAV Swarm in Operator-Guided Collective Behaviours', doi: '10.1109/ICRA48891.2023.10160630' },
    { venue: 'MPI 2023', title: 'Learning Methods with Range-Only Interactions in Active Systems', doi: '' },
];

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
        document.title = 'Services - Victor Retamal';

        this.container.innerHTML = `
            <div class="min-h-screen bg-dark-bg">
                <!-- Header -->
                <header class="bg-dark-surface/80 backdrop-blur-md border-b border-dark-border sticky top-0 z-40">
                    <div class="max-w-4xl mx-auto px-6 py-5">
                        <div class="flex justify-between items-center">
                            <div>
                                <h1 class="text-xl md:text-2xl font-bold text-white tracking-tight">Services</h1>
                                <p class="text-gray-400 text-sm">ML engineering, robotics, and consulting.</p>
                            </div>
                            <button id="back-btn" class="text-gray-400 hover:text-accent-cyan transition-all duration-300 flex items-center gap-2 text-sm">
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
                                </svg>
                                Home
                            </button>
                        </div>
                    </div>
                </header>

                <main class="max-w-4xl mx-auto px-6 pt-12 pb-12 space-y-10">

                    <!-- Hero -->
                    <section class="relative tut-animate">
                        <div class="text-center max-w-2xl mx-auto relative">
                            <div class="absolute -inset-16 bg-accent-cyan/[0.03] rounded-full blur-3xl -z-10"></div>
                            <h2 class="text-3xl md:text-5xl font-bold text-white mb-6 leading-tight tracking-tight">
                                Victor <span class="text-accent-cyan">Retamal</span>
                            </h2>
                            <p class="text-sm text-accent-cyan/80 font-medium tracking-wide uppercase mb-4">Senior Machine Learning Engineer</p>
                            <p class="text-gray-300 text-lg leading-relaxed mb-8 max-w-xl mx-auto">
                                ML and robotics engineer with 6+ years building sim-to-real pipelines, multi-agent systems,
                                computer vision, and medical imaging. 3 IEEE publications (RAL, ICRA) and a medical degree
                                that gives me a genuine edge in clinical AI.
                            </p>
                            <p class="text-gray-400 text-sm mb-8">
                                Available for contract work &middot; Remote or on-site (Spain/EU) &middot; EU citizen
                            </p>

                            <div class="flex flex-wrap justify-center gap-2.5 mb-10">
                                <span class="px-3.5 py-1.5 text-xs font-medium rounded-full border border-accent-cyan/20 text-accent-cyan/90 bg-accent-cyan/[0.05]">Gazebo/Isaac Lab/MuJoCo</span>
                                <span class="px-3.5 py-1.5 text-xs font-medium rounded-full border border-accent-cyan/20 text-accent-cyan/90 bg-accent-cyan/[0.05]">PyTorch &middot; JAX &middot; CUDA</span>
                                <span class="px-3.5 py-1.5 text-xs font-medium rounded-full border border-accent-cyan/20 text-accent-cyan/90 bg-accent-cyan/[0.05]">ROS/ROS2 &middot; PX4</span>
                                <span class="px-3.5 py-1.5 text-xs font-medium rounded-full border border-amber-400/20 text-amber-400/90 bg-amber-400/[0.05]">TensorRT &middot; ONNX &middot; Jetson</span>
                                <span class="px-3.5 py-1.5 text-xs font-medium rounded-full border border-accent-purple/20 text-accent-purple/90 bg-accent-purple/[0.05]">C/C++ &middot; Docker &middot; K8s</span>
                            </div>

                            <a href="#contact" class="btn-primary inline-flex items-center gap-2.5 text-base px-8 py-3.5 rounded-xl shadow-lg shadow-accent-cyan/20 hover:shadow-accent-cyan/30 transition-all duration-300">
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21.75 6.75v10.5a2.25 2.25 0 01-2.25 2.25h-15a2.25 2.25 0 01-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25m19.5 0v.243a2.25 2.25 0 01-1.07 1.916l-7.5 4.615a2.25 2.25 0 01-2.36 0L3.32 8.91a2.25 2.25 0 01-1.07-1.916V6.75" />
                                </svg>
                                Get in touch
                            </a>
                        </div>
                    </section>

                    <!-- Services -->
                    <section class="tut-animate relative">
                        <div class="absolute -inset-8 bg-accent-purple/[0.025] rounded-3xl blur-2xl -z-10"></div>
                        <div class="bg-dark-surface/50 backdrop-blur-sm border border-dark-border rounded-2xl p-8 md:p-10">
                            <h2 class="text-2xl font-bold text-white mb-8 tracking-tight">
                                What I <span class="text-accent-cyan">do</span>
                            </h2>
                            <div class="grid md:grid-cols-2 gap-x-12">
                                ${SERVICES.map(s => this.renderServiceItem(s)).join('')}
                            </div>
                        </div>
                    </section>

                    <!-- Publications -->
                    <section class="tut-animate">
                        <h2 class="text-2xl font-bold text-white mb-6 tracking-tight">
                            Publications
                        </h2>
                        <div class="space-y-3">
                            ${PUBLICATIONS.map(p => {
                                const tag = p.doi
                                    ? `<a href="https://doi.org/${p.doi}" target="_blank" rel="noopener noreferrer" class="block p-4 rounded-lg border border-dark-border hover:border-accent-purple/40 transition-colors group">`
                                    : `<div class="block p-4 rounded-lg border border-dark-border">`;
                                const close = p.doi ? '</a>' : '</div>';
                                return `
                                ${tag}
                                    <div class="flex items-start gap-3">
                                        <span class="text-xs font-mono text-accent-purple bg-accent-purple/10 px-2 py-0.5 rounded flex-shrink-0 mt-0.5">${p.venue}</span>
                                        <p class="text-sm text-gray-300 ${p.doi ? 'group-hover:text-white' : ''} transition-colors">${p.title}</p>
                                    </div>
                                ${close}`;
                            }).join('')}
                        </div>
                    </section>

                    <!-- Contact -->
                    <section id="contact" class="tut-animate relative rounded-2xl p-[1px]" style="background: linear-gradient(135deg, rgba(0,212,255,0.4), rgba(168,85,247,0.3), rgba(251,191,36,0.2), rgba(0,212,255,0.4))">
                        <div class="bg-dark-surface rounded-2xl p-8 md:p-10 relative overflow-hidden">
                            <div class="absolute top-0 right-0 w-72 h-72 bg-accent-cyan/[0.04] rounded-full blur-3xl"></div>
                            <div class="absolute bottom-0 left-0 w-56 h-56 bg-accent-purple/[0.03] rounded-full blur-3xl"></div>

                            <div class="relative text-center max-w-lg mx-auto">
                                <h2 class="text-2xl font-bold text-white mb-3 tracking-tight">
                                    Let's <span class="text-accent-cyan">work together</span>
                                </h2>
                                <p class="text-gray-400 text-sm mb-8 leading-relaxed">
                                    Available for contract work, consulting, and research collaborations. Tell me about your project.
                                </p>
                                <div class="flex flex-col sm:flex-row gap-3 justify-center">
                                    <a href="mailto:retamal1.victor@gmail.com?subject=Project%20Inquiry" class="btn-primary inline-flex items-center justify-center gap-2.5 text-base px-8 py-3.5 rounded-xl shadow-lg shadow-accent-cyan/20 hover:shadow-accent-cyan/30 transition-all duration-300">
                                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21.75 6.75v10.5a2.25 2.25 0 01-2.25 2.25h-15a2.25 2.25 0 01-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25m19.5 0v.243a2.25 2.25 0 01-1.07 1.916l-7.5 4.615a2.25 2.25 0 01-2.36 0L3.32 8.91a2.25 2.25 0 01-1.07-1.916V6.75" />
                                        </svg>
                                        Email me
                                    </a>
                                    <a href="https://www.linkedin.com/in/victor-retamal/" target="_blank" rel="noopener noreferrer" class="btn-secondary inline-flex items-center justify-center gap-2.5 text-base px-8 py-3.5 rounded-xl transition-all duration-300">
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
    }

    private renderServiceItem(service: Service): string {
        return `
            <div class="group flex items-start gap-3.5 py-3 px-3 -mx-3 rounded-xl transition-all duration-300 hover:bg-white/[0.02] min-h-[4rem]">
                <div class="w-10 h-10 rounded-lg bg-accent-cyan/[0.07] border border-accent-cyan/15 flex items-center justify-center flex-shrink-0 group-hover:bg-accent-cyan/[0.14] group-hover:border-accent-cyan/30 transition-all duration-300 mt-0.5">
                    <div class="text-accent-cyan/70 group-hover:text-accent-cyan transition-colors duration-300">${service.icon}</div>
                </div>
                <div>
                    <span class="text-white font-medium text-sm group-hover:text-accent-cyan transition-colors duration-300">${service.title}</span>
                    <p class="text-gray-500 group-hover:text-gray-400 text-sm mt-0.5 transition-colors duration-300 leading-snug">${service.description}</p>
                </div>
            </div>
        `;
    }

    private setupEventListeners(): void {
        const backBtn = document.getElementById('back-btn');
        backBtn?.addEventListener('click', () => {
            Navigation.toHome();
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
