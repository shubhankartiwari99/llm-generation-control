---
name: Obsidian Protocol
colors:
  surface: '#12121d'
  surface-dim: '#12121d'
  surface-bright: '#393844'
  surface-container-lowest: '#0d0d17'
  surface-container-low: '#1b1b25'
  surface-container: '#1f1f29'
  surface-container-high: '#292934'
  surface-container-highest: '#34343f'
  on-surface: '#e4e1f0'
  on-surface-variant: '#c7c4d8'
  inverse-surface: '#e4e1f0'
  inverse-on-surface: '#302f3b'
  outline: '#908fa1'
  outline-variant: '#464555'
  surface-tint: '#c1c1ff'
  primary: '#c1c1ff'
  on-primary: '#1400a8'
  primary-container: '#5c5cf4'
  on-primary-container: '#f9f5ff'
  inverse-primary: '#4847e0'
  secondary: '#b9c8de'
  on-secondary: '#233143'
  secondary-container: '#39485a'
  on-secondary-container: '#a7b6cc'
  tertiary: '#ffb68d'
  on-tertiary: '#532200'
  tertiary-container: '#b75400'
  on-tertiary-container: '#fff4f0'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e1dfff'
  primary-fixed-dim: '#c1c1ff'
  on-primary-fixed: '#09006b'
  on-primary-fixed-variant: '#2d27c9'
  secondary-fixed: '#d4e4fa'
  secondary-fixed-dim: '#b9c8de'
  on-secondary-fixed: '#0d1c2d'
  on-secondary-fixed-variant: '#39485a'
  tertiary-fixed: '#ffdbc9'
  tertiary-fixed-dim: '#ffb68d'
  on-tertiary-fixed: '#321200'
  on-tertiary-fixed-variant: '#763400'
  background: '#12121d'
  on-background: '#e4e1f0'
  surface-variant: '#34343f'
  surface-deep: '#07070C'
  surface-elevated: '#1A1A2E'
  border-subtle: '#2D2D44'
  text-primary: '#FFFFFF'
  text-secondary: '#94A3B8'
typography:
  headline-xl:
    fontFamily: Hanken Grotesk
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Hanken Grotesk
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Hanken Grotesk
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Hanken Grotesk
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Hanken Grotesk
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.05em
  headline-xl-mobile:
    fontFamily: Hanken Grotesk
    fontSize: 26px
    fontWeight: '700'
    lineHeight: 32px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  margin-desktop: 32px
  margin-mobile: 16px
  gutter: 24px
  unit: 4px
  stack-sm: 8px
  stack-md: 16px
  stack-lg: 24px
---

## Brand & Style
The design system is engineered for high-performance technical environments, specifically data-intensive dashboards and developer tools. The brand personality is precise, authoritative, and utilitarian, prioritizing cognitive clarity over decorative flair. 

The aesthetic follows a **Modern Corporate** direction with **Minimalist** influences. It utilizes a deep, "ink-trap" dark mode to reduce eye strain during prolonged sessions, punctuated by vibrant functional accents. The visual language conveys reliability through structured grids and high-contrast information density, ensuring that complex data remains the primary focus.

## Colors
This design system utilizes a dark-first palette to establish a technical "command center" atmosphere. 

- **Primary:** A high-vibrancy periwinkle blue used exclusively for primary actions, active states, and data highlights.
- **Neutral/Background:** The base is a deep navy-charcoal (`#0D0D17`), providing a sophisticated alternative to pure black.
- **Secondary/Text:** A cool slate gray is utilized for secondary information and iconography to maintain a clear visual hierarchy against the primary white headings.
- **Accents:** Use pure white (`#FFFFFF`) sparingly for high-priority text and "Surface Elevated" layers to create depth through value contrast rather than color.

## Typography
The typography strategy leverages two distinct families to separate "Intent" from "Data." 

**Hanken Grotesk** is used for all interface labels, headings, and body copy. It provides a sharp, contemporary professional look that remains legible at small sizes.

**JetBrains Mono** is reserved for technical strings, IDs, code snippets, and tabular data. This monospaced font signals to the user that they are looking at raw or system-generated information, enhancing the "technical dashboard" feel. 

High contrast is maintained by using White for headers and Slate Gray for metadata and labels.

## Layout & Spacing
The layout follows a **Fluid Grid** model with strict 4px increments. 

- **Desktop:** A 12-column grid with 24px gutters. Sidebars are fixed at 260px to maintain consistent navigation, while the main content area expands.
- **Information Density:** Components should use "compact" padding (e.g., 8px or 12px) to maximize the amount of data visible on screen without overcrowding.
- **Reflow:** On mobile, the 12-column grid collapses to a 4-column layout. Sidebars transform into bottom-drawer menus or hidden "hamburger" overlays.

## Elevation & Depth
In this design system, depth is communicated through **Tonal Layers** and **Low-contrast Outlines** rather than traditional shadows.

1.  **Background (Level 0):** `#07070C` - Used for the main canvas.
2.  **Surface (Level 1):** `#0D0D17` - Used for primary cards and content containers.
3.  **Elevated (Level 2):** `#1A1A2E` - Used for modals, dropdowns, and tooltips.

Surfaces are defined by 1px solid borders (`#2D2D44`). Shadows should be avoided or kept extremely subtle—only used for floating elements like modals, using a 0% blur, 12px offset deep-navy shadow to mimic a "stacked" physical appearance.

## Shapes
The shape language is "Soft" yet disciplined. A base corner radius of **4px (0.25rem)** is applied to buttons, input fields, and small components. This creates a technical, precise feel while avoiding the aggressive sharpness of pure 90-degree corners. 

Larger containers (Cards, Modals) utilize **8px (0.5rem)** to provide a slight visual distinction from interactive elements. Data visualizations and status chips may use a full pill-radius for distinct categorization.

## Components
- **Buttons:** Primary buttons use the `#5C5CF4` fill with white text. Ghost buttons use a `#2D2D44` border and slate text. Hover states should slightly brighten the fill.
- **Inputs:** Dark backgrounds (`#07070C`) with a 1px border. Focus state is a 1px `#5C5CF4` border with no outer glow.
- **Chips:** Monospaced text (JetBrains Mono) inside a subtle periwinkle or slate background with 20% opacity.
- **Cards:** No shadows. 1px border `#2D2D44` and a background of `#0D0D17`.
- **Lists:** Rows should be separated by 1px horizontal lines rather than alternating zebra stripes to maintain a clean, vertical rhythm.
- **Data Tables:** High-density, monospaced font for numbers, right-aligned. Headers should be all-caps slate gray.