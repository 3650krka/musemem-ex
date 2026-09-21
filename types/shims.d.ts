/**
 * Local typecheck shims. The real types come from the host Pi runtime
 * (@earendil-works/pi-coding-agent) which is NOT installed as a dependency
 * here — the package runs inside pi, where the peer exists. These minimal
 * declarations keep `tsc --noEmit` honest about our own usage without
 * pulling the whole coding-agent package into the sandbox.
 */

declare module "@earendil-works/pi-coding-agent" {
  export interface ExtensionAPI {
    on(event: string, handler: (event: any, ctx: any) => any): void;
    registerTool(definition: {
      name: string;
      label?: string;
      description: string;
      promptSnippet?: string;
      promptGuidelines?: string[];
      parameters: unknown;
      execute: (toolCallId: string, params: any, signal: AbortSignal, onUpdate: (update: unknown) => void, ctx: any) => Promise<unknown>;
    }): void;
    registerCommand(name: string, definition: { description: string; handler: (args: string | undefined, ctx: any) => Promise<void> | void }): void;
    /** Runtime model switch (used by the /memwalk into downshift gate). */
    setModel(model: unknown): Promise<boolean>;
  }
  export function getAgentDir(): string;
}
