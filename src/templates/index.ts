// Import all templates as raw strings at build time
import mainLayoutTemplate from './main-layout.html?raw';
import aboutSectionTemplate from './about-section.html?raw';
import researchSectionTemplate from './research-section.html?raw';
import projectsSectionTemplate from './projects-section.html?raw';
import resumeSectionTemplate from './resume-section.html?raw';
import footerTemplate from './footer.html?raw';
import heroTemplate from './hero.html?raw';
import headerTemplate from './header.html?raw';
import projectDetailTemplate from './project-detail.html?raw';
import projectNotFoundTemplate from './project-not-found.html?raw';
import recentPostsSectionTemplate from './recent-posts-section.html?raw';

// Export templates as a map
export const templates: Record<string, string> = {
    '/src/templates/main-layout.html': mainLayoutTemplate,
    '/src/templates/about-section.html': aboutSectionTemplate,
    '/src/templates/research-section.html': researchSectionTemplate,
    '/src/templates/projects-section.html': projectsSectionTemplate,
    '/src/templates/resume-section.html': resumeSectionTemplate,
    '/src/templates/footer.html': footerTemplate,
    '/src/templates/hero.html': heroTemplate,
    '/src/templates/header.html': headerTemplate,
    '/src/templates/project-detail.html': projectDetailTemplate,
    '/src/templates/project-not-found.html': projectNotFoundTemplate,
    '/src/templates/recent-posts-section.html': recentPostsSectionTemplate,
};

// Helper function to get template
export function getTemplate(path: string): string | undefined {
    return templates[path];
}