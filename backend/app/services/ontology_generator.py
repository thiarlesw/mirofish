"""
Ontology generation service: analyze text and produce entity/relationship type definitions for social simulation.
"""

import json
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient


ONTOLOGY_SYSTEM_PROMPT = """You are an expert knowledge-graph ontology designer. Your task is to analyze the given text and simulation requirements, and design entity types and relationship types suitable for **social-media opinion simulation**.

**Important: You must output valid JSON only, with no other content.**

## Context

We are building a **social-media opinion simulation system** where:
- Each entity is an "account" or actor that can post, interact, and spread information on social media
- Entities influence each other, repost, comment, and respond
- We need to simulate how different parties react and how information spreads in opinion events

Therefore **entities must be real-world actors that can speak and interact on social media**:

**Allowed**:
- Individuals (public figures, involved parties, opinion leaders, experts, ordinary people)
- Companies (including official accounts)
- Organizations (universities, associations, NGOs, unions, etc.)
- Government bodies, regulators
- Media (newspapers, TV, independent media, websites)
- Social platforms themselves
- Group representatives (alumni associations, fan clubs, advocacy groups, etc.)

**Not allowed**:
- Abstract concepts (e.g. "public opinion", "sentiment", "trend")
- Topics/themes (e.g. "academic integrity", "education reform")
- Stances (e.g. "supporters", "opponents")

## Output format

Output JSON with this structure:

```json
{
    "entity_types": [
        {
            "name": "EntityTypeName (English, PascalCase)",
            "description": "Short description (English, max 100 chars)",
            "attributes": [
                {
                    "name": "attribute_name (English, snake_case)",
                    "type": "text",
                    "description": "Attribute description"
                }
            ],
            "examples": ["Example entity 1", "Example entity 2"]
        }
    ],
    "edge_types": [
        {
            "name": "RELATIONSHIP_TYPE (English, UPPER_SNAKE_CASE)",
            "description": "Short description (English, max 100 chars)",
            "source_targets": [
                {"source": "Source entity type", "target": "Target entity type"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Brief analysis of the text content"
}
```

## Design rules (critical)

### 1. Entity types – strict

**Count: exactly 10 entity types.**

**Structure (both specific and fallback types required):**

Your 10 entity types must include:

A. **Fallback types (required; place last two in the list)**:
   - `Person`: Fallback for any individual. Use when a person does not fit a more specific person type.
   - `Organization`: Fallback for any organization. Use when an organization does not fit a more specific type.

B. **Specific types (8, designed from the text)**:
   - Design types for the main roles that appear in the text
   - E.g. for academic events: `Student`, `Professor`, `University`
   - E.g. for business events: `Company`, `CEO`, `Employee`

**Why fallbacks matter**: The text will mention varied people (e.g. teachers, anonymous users). Those without a matching specific type should map to `Person`. Similarly, small or ad-hoc groups map to `Organization`.

**Specific-type rules**:
- Identify frequently or crucially mentioned role types in the text
- Each specific type should have clear boundaries and avoid overlap
- description must clearly distinguish it from the fallback type

### 2. Relationship types

- Count: 6–10
- Relationships should reflect real social-media interactions
- source_targets must use only the entity types you define

### 3. Attributes

- 1–3 key attributes per entity type
- **Reserved names** (do not use): `name`, `uuid`, `group_id`, `created_at`, `summary`
- Prefer: `full_name`, `title`, `role`, `position`, `location`, `description`

## Entity type reference

**Individual (specific)**: Student, Professor, Journalist, Celebrity, Executive, Official, Lawyer, Doctor
**Individual (fallback)**: Person
**Organization (specific)**: University, Company, GovernmentAgency, MediaOutlet, Hospital, School, NGO
**Organization (fallback)**: Organization

## Relationship type reference

WORKS_FOR, STUDIES_AT, AFFILIATED_WITH, REPRESENTS, REGULATES, REPORTS_ON, COMMENTS_ON, RESPONDS_TO, SUPPORTS, OPPOSES, COLLABORATES_WITH, COMPETES_WITH
"""


class OntologyGenerator:
    """Analyzes text and generates entity/relationship type definitions for social simulation."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate ontology definition from documents and simulation requirement.

        Args:
            document_texts: List of document text chunks
            simulation_requirement: Description of simulation requirements
            additional_context: Optional extra context

        Returns:
            Ontology dict (entity_types, edge_types, analysis_summary, etc.)
        """
        user_message = self._build_user_message(
            document_texts,
            simulation_requirement,
            additional_context
        )
        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )
        result = self._validate_and_process(result)
        return result

    MAX_TEXT_LENGTH_FOR_LLM = 50000  # max chars sent to LLM for ontology analysis

    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Build the user prompt for ontology generation."""
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(original {original_length} chars; first {self.MAX_TEXT_LENGTH_FOR_LLM} used for ontology analysis)..."
        message = f"""## Simulation requirement

{simulation_requirement}

## Document content

{combined_text}
"""
        if additional_context:
            message += f"""
## Additional context

{additional_context}
"""
        message += """
Design entity types and relationship types suitable for social opinion simulation.

**Rules**:
1. Output exactly 10 entity types
2. Last two must be fallbacks: Person (individual) and Organization (organization)
3. First 8 are specific types derived from the text
4. All entity types must be real-world actors that can speak; no abstract concepts
5. Do not use reserved attribute names (name, uuid, group_id); use full_name, org_name, etc.
"""
        return message

    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and post-process LLM result (required fields, fallbacks, Zep limits)."""
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }
        
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)
        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                result["entity_types"] = result["entity_types"][:-to_remove]
            result["entity_types"].extend(fallbacks_to_add)
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]
        
        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """Convert ontology dict to Python code (ontology.py style)."""
        code_lines = [
            '"""',
            'Custom entity/edge type definitions, auto-generated by MiroFish for social opinion simulation.',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== Entity types ==============',
            '',
        ]
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        code_lines.append('# ============== Edge types ==============')
        code_lines.append('')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        code_lines.append('# ============== Type config ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)

