import os
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader