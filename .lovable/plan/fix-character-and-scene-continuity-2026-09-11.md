# Fix character and scene continuity

## Changes
- Rewrite prompt instructions so each timestamp keeps the established location and active cast unless the script explicitly changes them.
- Give every requested timestamp its immediate previous and next lines, enabling correct pronoun, action, and setting resolution.
- Enforce named characters from the current script line at render time using their pasted character-sheet traits, even if the prompt writer omitted them.
- Replace generic or copied fallback prompts with timestamp-specific prompts containing the current script line and nearby scene context.
- Keep the existing strict order: all timestamp prompts finish first, then exactly one image is generated per timestamp.

## Technical details
- Update prompt construction and deterministic fallback logic in the server-side storyboard writer.
- Pass each timestamp’s script text into final prompt composition and inject only directly named sheet characters.
- Update the browser’s emergency fallback so it never duplicates a neighboring timestamp’s prompt.
- Validate compilation and static behavior only; do not run paid image generation.
