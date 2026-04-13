export interface Section {
  id: string;
  title: string;
  element: HTMLElement;
}

export interface ThreeSceneConfig {
  containerId: string;
  modelPath?: string;
  cameraPosition?: [number, number, number];
  backgroundColor?: string;
  enableControls?: boolean;
}

export interface NavigationItem {
  label: string;
  href: string;
  external?: boolean;
}