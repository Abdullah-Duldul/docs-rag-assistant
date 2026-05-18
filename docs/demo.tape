# docs-rag-assistant demo — recorded with VHS
# https://github.com/charmbracelet/vhs

Output docs/demo.gif

Set Shell "zsh"
Set FontSize 14
Set Width 1400
Set Height 800
Set Theme "Catppuccin Mocha"
Set TypingSpeed 60ms
Set PlaybackSpeed 1.0

# Setup — navigate to project, clear screen, hidden from output
Hide
Type `cd "/Users/abdullah/Desktop/projects portfolio/docs-rag-assistant" && clear`
Enter
Sleep 2s
Show

# Scene 1 — Show the available commands
Type "uv run rag --help"
Enter
Sleep 3s

# Scene 2 — Cited answer (the money frame)
Type `uv run rag ask "What HTTP status code does the API return when rate limited?"`
Enter
Sleep 15s

# Scene 3 — Anti-hallucination demo
Type `uv run rag ask "Who is the CEO of Anthropic?"`
Enter
Sleep 12s

# Scene 4 — Evaluation harness (the closer)
Type "uv run rag eval"
Enter
Sleep 35s

# Final hold so the metrics stay on screen
Sleep 4s
