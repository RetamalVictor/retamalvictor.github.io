import * as yaml from 'js-yaml';

import demosYaml from '../data/demos.yaml?raw';

/**
 * One hero demo as a crawler should see it: the plain-language counterpart of
 * the "How it works" panel, which lives behind a button and a tab.
 */
export interface DemoSummary {
    id: string;
    name: string;
    summary: string;
    keywords: string[];
}

/** The demos in tab order. */
export const DEMOS: DemoSummary[] = (yaml.load(demosYaml) as { demos: DemoSummary[] }).demos;
