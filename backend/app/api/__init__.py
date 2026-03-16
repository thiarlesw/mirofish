"""
API route modules.
"""

from flask import Blueprint

graph_bp = Blueprint('graph', __name__)
simulation_bp = Blueprint('simulation', __name__)
report_bp = Blueprint('report', __name__)
from .graphiti_crud import graphiti_crud_bp  # noqa: E402

from . import graph  # noqa: E402, F401
from . import simulation  # noqa: E402, F401
from . import report  # noqa: E402, F401
from . import graphiti_crud  # noqa: E402, F401

