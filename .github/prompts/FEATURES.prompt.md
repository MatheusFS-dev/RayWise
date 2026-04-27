---
mode: 'agent'
---
Task: Generate a Project Features Documentation

1. Objective: Automatically search through all folders within the project directory and create a Markdown file (`FEATURES.md`) that documents each feature of the project. For each feature:
   - Provide a concise description of what the feature does.
   - Explain how it works at a high level (e.g., key components, workflows, or logic).
   - Mention any dependencies or related files.

2. Scope:
   - Include all relevant files and folders in the project.
   - Ignore non-code files (e.g., `.gitignore`, `.md` files, etc.) unless they are critical to understanding a feature.
   - Focus on source code, configuration files, and scripts.

3. Output Format:
   - Use a clear and structured Markdown format.
   - Group features by folder or module for better organization.
   - Example structure:
     ```markdown
     ## Feature Name
     - Description: [Brief description]
     - How it Works: [Explanation of the logic or workflow]
     - Example Usage: [How to use the feature, if applicable, add args if any or just a simple example]
     - Dependencies: [List of related files or libraries]
     ```

4. Assumptions:
   - Assume the project follows standard naming conventions and folder structures.
   - If a feature cannot be identified clearly, include a placeholder note (e.g., "Further clarification needed").

5. Deliverable:
   - A single `FEATURES.md` file at the root of the project directory.

6. Additional Notes:
   - Ensure the documentation is concise but informative.
   - Avoid technical jargon unless necessary.
   - If any ambiguity arises, flag it for review.

Start by scanning the project directory and identifying key features based on the folder and file structure. Then, generate the `FEATURES.md` file with the required details.