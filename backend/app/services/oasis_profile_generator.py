"""
OASIS Agent Profile generator: map Zep graph entities to OASIS simulation agent profiles.
Uses Zep retrieval to enrich context, detailed LLM persona generation, and distinguishes
individual vs group/organization entities.
"""

import json
import random
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI
from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('mirofish.oasis_profile')


@dataclass
class OasisAgentProfile:
    """OASIS agent profile data structure."""
    user_id: int
    user_name: str
    name: str
    bio: str
    persona: str
    karma: int = 1000
    friend_count: int = 100
    follower_count: int = 150
    statuses_count: int = 500
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None
    country: Optional[str] = None
    profession: Optional[str] = None
    interested_topics: List[str] = field(default_factory=list)
    source_entity_uuid: Optional[str] = None
    source_entity_type: Optional[str] = None
    
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    
    def to_reddit_format(self) -> Dict[str, Any]:
        """Convert to Reddit platform format (OASIS expects 'username')."""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "created_at": self.created_at,
        }
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics
        
        return profile
    
    def to_twitter_format(self) -> Dict[str, Any]:
        """Convert to Twitter platform format (OASIS expects 'username')."""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "created_at": self.created_at,
        }
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics
        
        return profile
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to full dict."""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "age": self.age,
            "gender": self.gender,
            "mbti": self.mbti,
            "country": self.country,
            "profession": self.profession,
            "interested_topics": self.interested_topics,
            "source_entity_uuid": self.source_entity_uuid,
            "source_entity_type": self.source_entity_type,
            "created_at": self.created_at,
        }


class OasisProfileGenerator:
    """
    OASIS profile generator: map Zep entities to agent profiles with Zep retrieval
    for richer context, detailed persona generation, and individual vs group handling.
    """
    MBTI_TYPES = [
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP"
    ]
    COUNTRIES = [
        "China", "US", "UK", "Japan", "Germany", "France",
        "Canada", "Australia", "Brazil", "India", "South Korea"
    ]
    INDIVIDUAL_ENTITY_TYPES = [
        "student", "alumni", "professor", "person", "publicfigure",
        "expert", "faculty", "official", "journalist", "activist"
    ]
    GROUP_ENTITY_TYPES = [
        "university", "governmentagency", "organization", "ngo", 
        "mediaoutlet", "company", "institution", "group", "community"
    ]
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        zep_api_key: Optional[str] = None,
        graph_id: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.zep_api_key = zep_api_key or Config.ZEP_API_KEY
        self.zep_client = None
        self.graph_id = graph_id
        if self.zep_api_key:
            try:
                self.zep_client = Zep(api_key=self.zep_api_key)
            except Exception as e:
                logger.warning(f"Zep client init failed: {e}")

    def generate_profile_from_entity(
        self,
        entity: EntityNode,
        user_id: int,
        use_llm: bool = True
    ) -> OasisAgentProfile:
        """Generate OASIS agent profile from a Zep entity."""
        entity_type = entity.get_entity_type() or "Entity"
        name = entity.name
        user_name = self._generate_username(name)
        context = self._build_entity_context(entity)
        if use_llm:
            profile_data = self._generate_profile_with_llm(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
                context=context
            )
        else:
            profile_data = self._generate_profile_rule_based(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes
            )
        
        return OasisAgentProfile(
            user_id=user_id,
            user_name=user_name,
            name=name,
            bio=profile_data.get("bio", f"{entity_type}: {name}"),
            persona=profile_data.get("persona", entity.summary or f"A {entity_type} named {name}."),
            karma=profile_data.get("karma", random.randint(500, 5000)),
            friend_count=profile_data.get("friend_count", random.randint(50, 500)),
            follower_count=profile_data.get("follower_count", random.randint(100, 1000)),
            statuses_count=profile_data.get("statuses_count", random.randint(100, 2000)),
            age=profile_data.get("age"),
            gender=profile_data.get("gender"),
            mbti=profile_data.get("mbti"),
            country=profile_data.get("country"),
            profession=profile_data.get("profession"),
            interested_topics=profile_data.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
        )
    
    def _generate_username(self, name: str) -> str:
        """Generate a username from entity name (sanitize + random suffix)."""
        username = name.lower().replace(" ", "_")
        username = ''.join(c for c in username if c.isalnum() or c == '_')
        suffix = random.randint(100, 999)
        return f"{username}_{suffix}"
    
    def _search_zep_for_entity(self, entity: EntityNode) -> Dict[str, Any]:
        """
        Fetch rich context for an entity via Zep graph search (edges + nodes in parallel).
        Returns dict with facts, node_summaries, context.
        """
        import concurrent.futures
        if not self.zep_client:
            return {"facts": [], "node_summaries": [], "context": ""}
        entity_name = entity.name
        results = {"facts": [], "node_summaries": [], "context": ""}
        if not self.graph_id:
            logger.debug("Skipping Zep retrieval: graph_id not set")
            return results
        comprehensive_query = f"All information, activities, events, relationships and background about {entity_name}"
        
        def search_edges():
            """Search edges (facts/relations) with retry."""
            max_retries = 3
            last_exception = None
            delay = 2.0
            
            for attempt in range(max_retries):
                try:
                    return self.zep_client.graph.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=30,
                        scope="edges",
                        reranker="rrf"
                    )
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.debug(f"Zep edge search attempt {attempt + 1} failed: {str(e)[:80]}, retrying...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(f"Zep edge search failed after {max_retries} attempts: {e}")
            return None

        def search_nodes():
            """Search nodes (entity summaries) with retry."""
            max_retries = 3
            last_exception = None
            delay = 2.0
            
            for attempt in range(max_retries):
                try:
                    return self.zep_client.graph.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=20,
                        scope="nodes",
                        reranker="rrf"
                    )
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.debug(f"Zep node search attempt {attempt + 1} failed: {str(e)[:80]}, retrying...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(f"Zep node search failed after {max_retries} attempts: {e}")
            return None
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                edge_future = executor.submit(search_edges)
                node_future = executor.submit(search_nodes)
                edge_result = edge_future.result(timeout=30)
                node_result = node_future.result(timeout=30)
            all_facts = set()
            if edge_result and hasattr(edge_result, 'edges') and edge_result.edges:
                for edge in edge_result.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        all_facts.add(edge.fact)
            results["facts"] = list(all_facts)
            
            all_summaries = set()
            if node_result and hasattr(node_result, 'nodes') and node_result.nodes:
                for node in node_result.nodes:
                    if hasattr(node, 'summary') and node.summary:
                        all_summaries.add(node.summary)
                    if hasattr(node, 'name') and node.name and node.name != entity_name:
                        all_summaries.add(f"Related entity: {node.name}")
            results["node_summaries"] = list(all_summaries)
            context_parts = []
            if results["facts"]:
                context_parts.append("Facts:\n" + "\n".join(f"- {f}" for f in results["facts"][:20]))
            if results["node_summaries"]:
                context_parts.append("Related entities:\n" + "\n".join(f"- {s}" for s in results["node_summaries"][:10]))
            results["context"] = "\n\n".join(context_parts)
            logger.info(f"Zep hybrid search done: {entity_name}, {len(results['facts'])} facts, {len(results['node_summaries'])} nodes")
        except concurrent.futures.TimeoutError:
            logger.warning(f"Zep retrieval timeout ({entity_name})")
        except Exception as e:
            logger.warning(f"Zep retrieval failed ({entity_name}): {e}")
        
        return results
    
    def _build_entity_context(self, entity: EntityNode) -> str:
        """Build full context: entity attributes, related edges/nodes, and Zep retrieval results."""
        context_parts = []
        if entity.attributes:
            attrs = []
            for key, value in entity.attributes.items():
                if value and str(value).strip():
                    attrs.append(f"- {key}: {value}")
            if attrs:
                context_parts.append("### Entity attributes\n" + "\n".join(attrs))
        existing_facts = set()
        if entity.related_edges:
            relationships = []
            for edge in entity.related_edges:
                fact = edge.get("fact", "")
                edge_name = edge.get("edge_name", "")
                direction = edge.get("direction", "")
                
                if fact:
                    relationships.append(f"- {fact}")
                    existing_facts.add(fact)
                elif edge_name:
                    if direction == "outgoing":
                        relationships.append(f"- {entity.name} --[{edge_name}]--> (related entity)")
                    else:
                        relationships.append(f"- (related entity) --[{edge_name}]--> {entity.name}")
            if relationships:
                context_parts.append("### Related facts and relationships\n" + "\n".join(relationships))
        if entity.related_nodes:
            related_info = []
            for node in entity.related_nodes:
                node_name = node.get("name", "")
                node_labels = node.get("labels", [])
                node_summary = node.get("summary", "")
                custom_labels = [l for l in node_labels if l not in ["Entity", "Node"]]
                label_str = f" ({', '.join(custom_labels)})" if custom_labels else ""
                
                if node_summary:
                    related_info.append(f"- **{node_name}**{label_str}: {node_summary}")
                else:
                    related_info.append(f"- **{node_name}**{label_str}")
            
            if related_info:
                context_parts.append("### Related entity info\n" + "\n".join(related_info))
        zep_results = self._search_zep_for_entity(entity)
        if zep_results.get("facts"):
            new_facts = [f for f in zep_results["facts"] if f not in existing_facts]
            if new_facts:
                context_parts.append("### Facts from Zep retrieval\n" + "\n".join(f"- {f}" for f in new_facts[:15]))
        if zep_results.get("node_summaries"):
            context_parts.append("### Related nodes from Zep retrieval\n" + "\n".join(f"- {s}" for s in zep_results["node_summaries"][:10]))
        
        return "\n\n".join(context_parts)
    
    def _is_individual_entity(self, entity_type: str) -> bool:
        """Whether this entity type is an individual (vs group/org)."""
        return entity_type.lower() in self.INDIVIDUAL_ENTITY_TYPES

    def _is_group_entity(self, entity_type: str) -> bool:
        """Whether this entity type is a group/organization."""
        return entity_type.lower() in self.GROUP_ENTITY_TYPES

    def _generate_profile_with_llm(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> Dict[str, Any]:
        """Generate detailed persona via LLM; individual vs group/org prompts."""
        is_individual = self._is_individual_entity(entity_type)
        
        if is_individual:
            prompt = self._build_individual_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )
        else:
            prompt = self._build_group_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )

        max_attempts = 3
        last_error = None
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(is_individual)},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)
                )
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'length':
                    logger.warning(f"LLM output truncated (attempt {attempt+1}), fixing...")
                    content = self._fix_truncated_json(content)
                try:
                    result = json.loads(content)
                    if "bio" not in result or not result["bio"]:
                        result["bio"] = entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}"
                    if "persona" not in result or not result["persona"]:
                        result["persona"] = entity_summary or f"{entity_name} is a {entity_type}."
                    return result
                except json.JSONDecodeError as je:
                    logger.warning(f"JSON parse failed (attempt {attempt+1}): {str(je)[:80]}")
                    result = self._try_fix_json(content, entity_name, entity_type, entity_summary)
                    if result.get("_fixed"):
                        del result["_fixed"]
                        return result
                    
                    last_error = je
                    
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(1 * (attempt + 1))
        logger.warning(f"LLM persona generation failed after {max_attempts} attempts: {last_error}, using rule-based")
        return self._generate_profile_rule_based(
            entity_name, entity_type, entity_summary, entity_attributes
        )
    
    def _fix_truncated_json(self, content: str) -> str:
        """Try to close truncated JSON (e.g. cut by max_tokens)."""
        import re
        content = content.strip()
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        if content and content[-1] not in '",}]':
            content += '"'
        content += ']' * open_brackets
        content += '}' * open_braces
        
        return content
    
    def _try_fix_json(self, content: str, entity_name: str, entity_type: str, entity_summary: str = "") -> Dict[str, Any]:
        """Try to repair malformed JSON and extract bio/persona."""
        import re
        content = self._fix_truncated_json(content)
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            def fix_string_newlines(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string_newlines, json_str)
            try:
                result = json.loads(json_str)
                result["_fixed"] = True
                return result
            except json.JSONDecodeError:
                try:
                    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                    json_str = re.sub(r'\s+', ' ', json_str)
                    result = json.loads(json_str)
                    result["_fixed"] = True
                    return result
                except Exception:
                    pass
        bio_match = re.search(r'"bio"\s*:\s*"([^"]*)"', content)
        persona_match = re.search(r'"persona"\s*:\s*"([^"]*)', content)
        bio = bio_match.group(1) if bio_match else (entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}")
        persona = persona_match.group(1) if persona_match else (entity_summary or f"{entity_name} is a {entity_type}.")
        if bio_match or persona_match:
            logger.info("Extracted partial info from malformed JSON")
            return {"bio": bio, "persona": persona, "_fixed": True}
        logger.warning("JSON repair failed, returning fallback structure")
        return {
            "bio": entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}",
            "persona": entity_summary or f"{entity_name} is a {entity_type}."
        }
    
    def _get_system_prompt(self, is_individual: bool) -> str:
        """System prompt for persona generation."""
        return (
            "You are an expert at generating detailed, realistic social media user personas "
            "for opinion simulation. Return valid JSON only; no unescaped newlines in string values. "
            "Respond in Chinese."
        )

    def _build_individual_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Build prompt for individual entity persona."""
        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "None"
        context_str = context[:3000] if context else "No extra context"
        return f"""Generate a detailed social media user persona for this entity.

Entity name: {entity_name}
Entity type: {entity_type}
Entity summary: {entity_summary}
Entity attributes: {attrs_str}

Context:
{context_str}

Return JSON with these fields:
1. bio: short bio (about 200 chars)
2. persona: detailed persona text (~2000 chars) covering: basic info (age, job, education, location), background (key experiences, relation to events, social ties), personality (MBTI, traits, expression), social media behavior (posting frequency, content preferences, interaction style), stance and views, distinctive traits, personal memory (this individual's link to the event and past actions/reactions).
3. age: integer
4. gender: "male" or "female"
5. mbti: e.g. INTJ, ENFP
6. country: e.g. China (use Chinese for country name)
7. profession: string
8. interested_topics: array of strings

Rules: all values must be strings or numbers, no newlines; persona must be one coherent paragraph; use Chinese except gender must be "male"/"female"; age must be integer, gender must be "male" or "female".
"""

    def _build_group_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Build prompt for group/organization entity persona."""
        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "None"
        context_str = context[:3000] if context else "No extra context"
        return f"""Generate a detailed social media account persona for this organization/group entity.

Entity name: {entity_name}
Entity type: {entity_type}
Entity summary: {entity_summary}
Entity attributes: {attrs_str}

Context:
{context_str}

Return JSON with: 1. bio: official account bio (~200 chars). 2. persona: detailed account persona (~2000 chars) covering: org basics (name, nature, founding, main functions), account positioning (type, audience, core role), tone and style, content and posting habits, stance and controversy handling, represented group, org memory (link to event and past actions/reactions). 3. age: 30. 4. gender: "other". 5. mbti: e.g. ISTJ for formal style. 6. country: use Chinese. 7. profession: org role. 8. interested_topics: array.

Rules: no nulls; persona one coherent paragraph, no newlines; use Chinese except gender="other"; age=30, gender="other"; tone must match org identity."""
    
    def _generate_profile_rule_based(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate basic persona by rules (no LLM)."""
        entity_type_lower = entity_type.lower()
        
        if entity_type_lower in ["student", "alumni"]:
            return {
                "bio": f"{entity_type} with interests in academics and social issues.",
                "persona": f"{entity_name} is a {entity_type.lower()} who is actively engaged in academic and social discussions. They enjoy sharing perspectives and connecting with peers.",
                "age": random.randint(18, 30),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": "Student",
                "interested_topics": ["Education", "Social Issues", "Technology"],
            }
        
        elif entity_type_lower in ["publicfigure", "expert", "faculty"]:
            return {
                "bio": f"Expert and thought leader in their field.",
                "persona": f"{entity_name} is a recognized {entity_type.lower()} who shares insights and opinions on important matters. They are known for their expertise and influence in public discourse.",
                "age": random.randint(35, 60),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_attributes.get("occupation", "Expert"),
                "interested_topics": ["Politics", "Economics", "Culture & Society"],
            }
        
        elif entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
            return {
                "bio": f"Official account for {entity_name}. News and updates.",
                "persona": f"{entity_name} is a media entity that reports news and facilitates public discourse. The account shares timely updates and engages with the audience on current events.",
                "age": 30,
                "gender": "other",
                "mbti": "ISTJ",
                "country": "China",
                "profession": "Media",
                "interested_topics": ["General News", "Current Events", "Public Affairs"],
            }
        
        elif entity_type_lower in ["university", "governmentagency", "ngo", "organization"]:
            return {
                "bio": f"Official account of {entity_name}.",
                "persona": f"{entity_name} is an institutional entity that communicates official positions, announcements, and engages with stakeholders on relevant matters.",
                "age": 30,
                "gender": "other",
                "mbti": "ISTJ",
                "country": "China",
                "profession": entity_type,
                "interested_topics": ["Public Policy", "Community", "Official Announcements"],
            }
        else:
            return {
                "bio": entity_summary[:150] if entity_summary else f"{entity_type}: {entity_name}",
                "persona": entity_summary or f"{entity_name} is a {entity_type.lower()} participating in social discussions.",
                "age": random.randint(25, 50),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_type,
                "interested_topics": ["General", "Social Issues"],
            }
    
    def set_graph_id(self, graph_id: str):
        """Set graph ID for Zep retrieval."""
        self.graph_id = graph_id

    def generate_profiles_from_entities(
        self,
        entities: List[EntityNode],
        use_llm: bool = True,
        progress_callback: Optional[callable] = None,
        graph_id: Optional[str] = None,
        parallel_count: int = 5,
        realtime_output_path: Optional[str] = None,
        output_platform: str = "reddit"
    ) -> List[OasisAgentProfile]:
        """Batch-generate agent profiles from entities (parallel); optional realtime file write."""
        import concurrent.futures
        from threading import Lock
        if graph_id:
            self.graph_id = graph_id
        total = len(entities)
        profiles = [None] * total
        completed_count = [0]
        lock = Lock()

        def save_profiles_realtime():
            """Write currently generated profiles to file."""
            if not realtime_output_path:
                return
            with lock:
                existing_profiles = [p for p in profiles if p is not None]
                if not existing_profiles:
                    return
                
                try:
                    if output_platform == "reddit":
                        profiles_data = [p.to_reddit_format() for p in existing_profiles]
                        with open(realtime_output_path, 'w', encoding='utf-8') as f:
                            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                    else:
                        import csv
                        profiles_data = [p.to_twitter_format() for p in existing_profiles]
                        if profiles_data:
                            fieldnames = list(profiles_data[0].keys())
                            with open(realtime_output_path, 'w', encoding='utf-8', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(profiles_data)
                except Exception as e:
                    logger.warning(f"Realtime save profiles failed: {e}")

        def generate_single_profile(idx: int, entity: EntityNode) -> tuple:
            """Worker: generate one profile for an entity."""
            entity_type = entity.get_entity_type() or "Entity"
            try:
                profile = self.generate_profile_from_entity(
                    entity=entity,
                    user_id=idx,
                    use_llm=use_llm
                )
                self._print_generated_profile(entity.name, entity_type, profile)
                return idx, profile, None
            except Exception as e:
                logger.error(f"Profile generation failed for entity {entity.name}: {str(e)}")
                fallback_profile = OasisAgentProfile(
                    user_id=idx,
                    user_name=self._generate_username(entity.name),
                    name=entity.name,
                    bio=f"{entity_type}: {entity.name}",
                    persona=entity.summary or f"A participant in social discussions.",
                    source_entity_uuid=entity.uuid,
                    source_entity_type=entity_type,
                )
                return idx, fallback_profile, str(e)
        
        logger.info(f"Starting parallel profile generation: {total} entities, workers={parallel_count}")
        print(f"\n{'='*60}")
        print(f"Generating agent profiles - {total} entities, parallel={parallel_count}")
        print(f"{'='*60}\n")
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_count) as executor:
            future_to_entity = {
                executor.submit(generate_single_profile, idx, entity): (idx, entity)
                for idx, entity in enumerate(entities)
            }
            
            for future in concurrent.futures.as_completed(future_to_entity):
                idx, entity = future_to_entity[future]
                entity_type = entity.get_entity_type() or "Entity"
                
                try:
                    result_idx, profile, error = future.result()
                    profiles[result_idx] = profile
                    
                    with lock:
                        completed_count[0] += 1
                        current = completed_count[0]
                    
                    save_profiles_realtime()
                    if progress_callback:
                        progress_callback(current, total, f"Done {current}/{total}: {entity.name} ({entity_type})")
                    if error:
                        logger.warning(f"[{current}/{total}] {entity.name} using fallback: {error}")
                    else:
                        logger.info(f"[{current}/{total}] Profile generated: {entity.name} ({entity_type})")
                except Exception as e:
                    logger.error(f"Exception processing entity {entity.name}: {str(e)}")
                    with lock:
                        completed_count[0] += 1
                    profiles[idx] = OasisAgentProfile(
                        user_id=idx,
                        user_name=self._generate_username(entity.name),
                        name=entity.name,
                        bio=f"{entity_type}: {entity.name}",
                        persona=entity.summary or "A participant in social discussions.",
                        source_entity_uuid=entity.uuid,
                        source_entity_type=entity_type,
                    )
                    save_profiles_realtime()
        print(f"\n{'='*60}")
        print(f"Profile generation done. Total: {len([p for p in profiles if p])} agents")
        print(f"{'='*60}\n")
        return profiles

    def _print_generated_profile(self, entity_name: str, entity_type: str, profile: OasisAgentProfile):
        """Print generated profile to console (full content)."""
        separator = "-" * 70
        topics_str = ', '.join(profile.interested_topics) if profile.interested_topics else 'None'
        output_lines = [
            f"\n{separator}",
            f"[Generated] {entity_name} ({entity_type})",
            f"{separator}",
            f"Username: {profile.user_name}",
            f"",
            f"[Bio]",
            f"{profile.bio}",
            f"",
            f"[Persona]",
            f"{profile.persona}",
            f"",
            f"[Attributes]",
            f"Age: {profile.age} | Gender: {profile.gender} | MBTI: {profile.mbti}",
            f"Profession: {profile.profession} | Country: {profile.country}",
            f"Topics: {topics_str}",
            separator
        ]
        print("\n".join(output_lines))

    def save_profiles(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """Save profiles to file. OASIS: Twitter=CSV, Reddit=JSON."""
        if platform == "twitter":
            self._save_twitter_csv(profiles, file_path)
        else:
            self._save_reddit_json(profiles, file_path)
    
    def _save_twitter_csv(self, profiles: List[OasisAgentProfile], file_path: str):
        """Save Twitter profiles as CSV (OASIS: user_id, name, username, user_char, description)."""
        import csv
        if not file_path.endswith('.csv'):
            file_path = file_path.replace('.json', '.csv')
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            headers = ['user_id', 'name', 'username', 'user_char', 'description']
            writer.writerow(headers)
            for idx, profile in enumerate(profiles):
                user_char = profile.bio
                if profile.persona and profile.persona != profile.bio:
                    user_char = f"{profile.bio} {profile.persona}"
                user_char = user_char.replace('\n', ' ').replace('\r', ' ')
                description = profile.bio.replace('\n', ' ').replace('\r', ' ')
                row = [idx, profile.name, profile.user_name, user_char, description]
                writer.writerow(row)
        logger.info(f"Saved {len(profiles)} Twitter profiles to {file_path} (OASIS CSV)")

    def _normalize_gender(self, gender: Optional[str]) -> str:
        """Normalize gender to OASIS values: male, female, other (incl. Chinese mapping)."""
        if not gender:
            return "other"
        gender_lower = gender.lower().strip()
        gender_map = {
            "男": "male",
            "女": "female",
            "机构": "other",
            "其他": "other",
            "male": "male",
            "female": "female",
            "other": "other",
        }
        return gender_map.get(gender_lower, "other")

    def _save_reddit_json(self, profiles: List[OasisAgentProfile], file_path: str):
        """Save Reddit profiles as JSON (same shape as to_reddit_format(); user_id required for OASIS)."""
        data = []
        for idx, profile in enumerate(profiles):
            item = {
                "user_id": profile.user_id if profile.user_id is not None else idx,
                "username": profile.user_name,
                "name": profile.name,
                "bio": profile.bio[:150] if profile.bio else f"{profile.name}",
                "persona": profile.persona or f"{profile.name} is a participant in social discussions.",
                "karma": profile.karma if profile.karma else 1000,
                "created_at": profile.created_at,
                "age": profile.age if profile.age else 30,
                "gender": self._normalize_gender(profile.gender),
                "mbti": profile.mbti if profile.mbti else "ISTJ",
                "country": profile.country if profile.country else "China",
            }
            if profile.profession:
                item["profession"] = profile.profession
            if profile.interested_topics:
                item["interested_topics"] = profile.interested_topics
            data.append(item)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(profiles)} Reddit profiles to {file_path} (JSON with user_id)")

    def save_profiles_to_json(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """[Deprecated] Use save_profiles() instead."""
        logger.warning("save_profiles_to_json is deprecated, use save_profiles")
        self.save_profiles(profiles, file_path, platform)

