## Testing

- Required PowerShell: 5.1
- Required Pester: 5.7.1

Run from repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_tests.ps1
```

Note:
- `TestRegistry` is disabled in the runner due to restricted registry environments.

