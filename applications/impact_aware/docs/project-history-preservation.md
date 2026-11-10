# MTech Project: History Preservation Plan

## Project evolution

The Impact-Aware Data Drift implementation and the later Intelligent
Predictive Analytics implementation represent successive stages of
the same MTech project.

The intended final deliverable is one unified machine-learning platform.

## Preservation requirements

During future integration:

- Preserve original Git commit hashes.
- Preserve original author and committer dates.
- Preserve commit messages and parent relationships.
- Retain original source files and application functionality.
- Preserve dashboards, tests, configurations and documentation.
- Back up untracked and ignored files separately.
- Keep the original repositories available and unchanged.

## Integration validation

Before accepting the unified repository:

1. Compare original and imported commit histories.
2. Verify original commit reachability.
3. Compare source-file inventories.
4. Verify application and dashboard functionality.
5. Run available automated tests.
6. Document any intentional architectural changes.

The original repositories must not be rewritten or force-pushed
during integration.
