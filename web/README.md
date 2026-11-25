# MapleMind web app

React (Vite) UI + Node.js (Express) gateway over the Python LangGraph agents. See the main README, §17b.

    cd client && npm install && npm run build
    cd ../server && npm install && npm start      # http://localhost:8787

Environment: `PORT` (default 8787), `PYTHON` (default python3). All `MAPLEMIND__*` variables pass through to the agents,
e.g. `MAPLEMIND__LLM__PROVIDER=ollama npm start`.
