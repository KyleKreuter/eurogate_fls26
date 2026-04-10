# =============================================================================
# Eurogate FLS26 — Dashboard Developer Makefile
#
# Quick start:
#   make install    # one-time: install backend + web deps
#   make db         # one-time: reassemble reefer.db from zip parts
#   make dev        # start backend + frontend together (Ctrl-C stops both)
#
# Individual targets:
#   make backend    # uvicorn only      → http://127.0.0.1:8080
#   make web        # Vite dev only     → http://127.0.0.1:5173
#   make build      # production build of web/
#   make stop       # kill any running dev servers
#   make lint       # tsc + eslint on web/
#   make help       # print this overview
# =============================================================================

# Absolute paths so targets work from any subdirectory
ROOT    := $(CURDIR)
VENV    := $(ROOT)/.venv
PYTHON  := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn
BACKEND := $(ROOT)/backend
WEB     := $(ROOT)/web

# Colors for friendly output
CYAN  := \033[36m
DIM   := \033[2m
BOLD  := \033[1m
RESET := \033[0m

.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------

.PHONY: help
help:
	@printf "$(BOLD)Eurogate FLS26 — dashboard dev targets$(RESET)\n"
	@printf "\n"
	@printf "  $(CYAN)make install$(RESET)   Install backend venv + web npm deps\n"
	@printf "  $(CYAN)make db$(RESET)        Reassemble reefer.db from zip parts (~1.1GB)\n"
	@printf "  $(CYAN)make dev$(RESET)       Start backend (:8080) + Vite (:5173) in parallel\n"
	@printf "  $(CYAN)make backend$(RESET)   Backend only (FastAPI + uvicorn --reload)\n"
	@printf "  $(CYAN)make web$(RESET)       Vite dev server only\n"
	@printf "  $(CYAN)make build$(RESET)     Production build of web/dist\n"
	@printf "  $(CYAN)make stop$(RESET)      Kill uvicorn + vite processes\n"
	@printf "  $(CYAN)make lint$(RESET)      tsc -b + eslint on web/\n"
	@printf "\n"
	@printf "$(DIM)Prereqs: Python 3.12+, Node 20+, git-lfs (for participant_package data)$(RESET)\n"

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------

.PHONY: install
install: install-backend install-web
	@printf "$(BOLD)✓ All dependencies installed.$(RESET) Run $(CYAN)make dev$(RESET) to start.\n"

.PHONY: install-backend
install-backend:
	@if [ ! -d "$(VENV)" ]; then \
		printf "→ Creating Python venv at $(VENV)\n"; \
		python3 -m venv $(VENV); \
	fi
	@printf "→ Installing backend dependencies\n"
	@$(PIP) install --quiet --upgrade pip
	@$(PIP) install --quiet \
		"fastapi>=0.115.0" \
		"uvicorn[standard]>=0.32.0" \
		"pydantic>=2.9.0" \
		"pydantic-settings>=2.6.0"
	@printf "✓ Backend deps ready ($(VENV))\n"

.PHONY: install-web
install-web:
	@printf "→ Installing web dependencies\n"
	@cd $(WEB) && npm install --silent
	@printf "✓ Web deps ready (web/node_modules)\n"

# -----------------------------------------------------------------------------
# Database bootstrap
# -----------------------------------------------------------------------------

.PHONY: db
db:
	@if [ -f "$(BACKEND)/reefer.db" ]; then \
		printf "✓ $(BACKEND)/reefer.db already exists ($$(du -h $(BACKEND)/reefer.db | cut -f1))\n"; \
	else \
		printf "→ Reassembling reefer.db from 4 zip parts…\n"; \
		cd $(BACKEND) && cat reefer.db.zip.0* > reefer.db.zip && unzip -o reefer.db.zip && rm reefer.db.zip; \
		printf "✓ $(BACKEND)/reefer.db ready ($$(du -h $(BACKEND)/reefer.db | cut -f1))\n"; \
	fi

# -----------------------------------------------------------------------------
# Dev servers
# -----------------------------------------------------------------------------

.PHONY: backend
backend: db
	@printf "$(BOLD)→ Starting FastAPI backend on http://127.0.0.1:8080$(RESET)\n"
	@printf "$(DIM)  API docs: http://127.0.0.1:8080/docs$(RESET)\n"
	@cd $(BACKEND) && $(UVICORN) app.main:app \
		--host 127.0.0.1 --port 8080 --reload

.PHONY: web
web:
	@printf "$(BOLD)→ Starting Vite dev server on http://127.0.0.1:5173$(RESET)\n"
	@cd $(WEB) && npm run dev -- --host 127.0.0.1

.PHONY: dev
dev: db
	@printf "$(BOLD)→ Starting backend + web dev servers$(RESET)\n"
	@printf "   $(CYAN)backend$(RESET)  http://127.0.0.1:8080   $(DIM)(FastAPI auto-docs: /docs)$(RESET)\n"
	@printf "   $(CYAN)web$(RESET)      http://127.0.0.1:5173   $(DIM)(Vite HMR + proxy /api → :8080)$(RESET)\n"
	@printf "$(DIM)   Press Ctrl-C once to stop both.$(RESET)\n\n"
	@trap 'printf "\n$(BOLD)→ Shutting down…$(RESET)\n"; kill 0 2>/dev/null; exit 0' INT TERM EXIT; \
	( cd $(BACKEND) && $(UVICORN) app.main:app --host 127.0.0.1 --port 8080 --reload 2>&1 | sed 's/^/[backend] /' ) & \
	( cd $(WEB) && npm run dev -- --host 127.0.0.1 2>&1 | sed 's/^/[web]     /' ) & \
	wait

# -----------------------------------------------------------------------------
# Build + lint + housekeeping
# -----------------------------------------------------------------------------

.PHONY: build
build:
	@printf "$(BOLD)→ Building web/dist$(RESET)\n"
	@cd $(WEB) && npm run build
	@printf "✓ Build complete: web/dist/\n"

.PHONY: lint
lint:
	@printf "$(BOLD)→ TypeScript + eslint$(RESET)\n"
	@cd $(WEB) && npx tsc -b && npm run lint

.PHONY: stop
stop:
	@printf "→ Killing uvicorn + vite processes…\n"
	-@pkill -f "uvicorn app.main:app" 2>/dev/null || true
	-@pkill -f "vite" 2>/dev/null || true
	@printf "✓ Stopped\n"

.PHONY: clean
clean: stop
	@printf "→ Removing web/dist\n"
	-@rm -rf $(WEB)/dist
	@printf "✓ Cleaned (reefer.db + node_modules kept)\n"
