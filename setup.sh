#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot setup script for the HR Policy RAG System
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# =============================================================================

set -e  # Exit on any error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════╗"
echo "║     HR Policy RAG System — Setup               ║"
echo "╚════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── 1. Python version check ────────────────────────────────────────────────
echo -e "${BLUE}[1/6] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED="3.10"
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo -e "  ${GREEN}✓ Python $PYTHON_VERSION${NC}"
else
    echo -e "  ${RED}✗ Python 3.10+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

# ── 2. Virtual environment ─────────────────────────────────────────────────
echo -e "${BLUE}[2/6] Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "  ${GREEN}✓ Created venv${NC}"
else
    echo -e "  ${YELLOW}→ venv already exists, skipping${NC}"
fi

source venv/bin/activate
echo -e "  ${GREEN}✓ Activated venv${NC}"

# ── 3. Install dependencies ────────────────────────────────────────────────
echo -e "${BLUE}[3/6] Installing dependencies (this may take a few minutes)...${NC}"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo -e "  ${GREEN}✓ Dependencies installed${NC}"

# ── 4. Environment file ────────────────────────────────────────────────────
echo -e "${BLUE}[4/6] Configuring environment...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "  ${YELLOW}→ Created .env from template${NC}"
    echo -e "  ${YELLOW}  ⚠  EDIT .env and add your GROQ_API_KEY before continuing${NC}"
    echo -e "     Get a free key at: ${BLUE}https://console.groq.com${NC}"
else
    echo -e "  ${GREEN}✓ .env already exists${NC}"
fi

# ── 5. Documents directory ─────────────────────────────────────────────────
echo -e "${BLUE}[5/6] Checking documents directory...${NC}"
mkdir -p documents
DOC_COUNT=$(find documents -type f \( -name "*.pdf" -o -name "*.docx" -o -name "*.txt" \) 2>/dev/null | wc -l)
if [ "$DOC_COUNT" -eq 0 ]; then
    echo -e "  ${YELLOW}→ 'documents/' is empty${NC}"
    echo -e "     Add your HR policy PDFs/DOCX files to ./documents/"
    echo -e "     Then run: ${BLUE}python ingestion.py${NC}"
else
    echo -e "  ${GREEN}✓ Found $DOC_COUNT document(s) in documents/${NC}"
fi

# ── 6. Done ────────────────────────────────────────────────────────────────
echo -e "${BLUE}[6/6] Setup complete!${NC}"
echo ""
echo -e "${GREEN}══════════════════════════════════════${NC}"
echo -e "${GREEN}  Next steps:${NC}"
echo -e "${GREEN}══════════════════════════════════════${NC}"
echo ""
echo -e "  ${YELLOW}1.${NC} Edit ${BLUE}.env${NC} — add your GROQ_API_KEY"
echo -e "  ${YELLOW}2.${NC} Copy policy PDFs → ${BLUE}./documents/${NC}"
echo -e "  ${YELLOW}3.${NC} Run ingestion: ${BLUE}python ingestion.py${NC}"
echo -e "  ${YELLOW}4.${NC} Start API:     ${BLUE}uvicorn api:app --reload --port 8000${NC}"
echo -e "  ${YELLOW}5.${NC} Open frontend: ${BLUE}./frontend/index.html${NC} in browser"
echo -e "       Or use VS Code Live Server / any static server"
echo ""
echo -e "  ${YELLOW}CLI mode:${NC} ${BLUE}python rag_pipeline.py${NC}"
echo ""
