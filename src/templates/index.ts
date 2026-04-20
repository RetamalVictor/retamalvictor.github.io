// Import all templates as raw strings at build time
import mainLayoutTemplate from './main-layout.html?raw';
import aboutSectionTemplate from './about-section.html?raw';
import researchSectionTemplate from './research-section.html?raw';
import footerTemplate from './footer.html?raw';
import heroTemplate from './hero.html?raw';
import headerTemplate from './header.html?raw';
import selectedWorkSectionTemplate from './selected-work-section.html?raw';
import publicationsSectionTemplate from './publications-section.html?raw';
import recentPostsSectionTemplate from './recent-posts-section.html?raw';

// Export templates as a map
export const templates: Record<string, string> = {
    '/src/templates/main-layout.html': mainLayoutTemplate,
    '/src/templates/about-section.html': aboutSectionTemplate,
    '/src/templates/research-section.html': researchSectionTemplate,
    '/src/templates/footer.html': footerTemplate,
    '/src/templates/hero.html': heroTemplate,
    '/src/templates/header.html': headerTemplate,
    '/src/templates/selected-work-section.html': selectedWorkSectionTemplate,
    '/src/templates/publications-section.html': publicationsSectionTemplate,
    '/src/templates/recent-posts-section.html': recentPostsSectionTemplate,
};

// Helper function to get template
export function getTemplate(path: string): string | undefined {
    return templates[path];
}