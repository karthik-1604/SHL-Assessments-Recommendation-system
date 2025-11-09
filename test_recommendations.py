#!/usr/bin/env python3
"""
Test the recommendation system to identify quality issues
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.recommender import SHLRecommender

def test_queries():
    """Test various queries to see recommendation quality"""

    print("🚀 Initializing SHL Recommender...")
    recommender = SHLRecommender()

    if not recommender.initialize():
        print("❌ Failed to initialize")
        return

    test_queries = [
        "Java developer with good communication skills",
        "Python programmer who can collaborate with business teams",
        "Senior software engineer with leadership abilities",
        "Customer service representative",
        "Data analyst with SQL skills",
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)

        result = recommender.recommend(query)
        assessments = result.get('recommended_assessments', [])

        # Analyze the results
        technical_count = 0
        behavioral_count = 0

        for i, assessment in enumerate(assessments[:10], 1):
            test_types = assessment['test_type']
            test_types_str = ', '.join(test_types)

            # Count technical vs behavioral
            if any('Knowledge' in t or 'Skills' in t or 'Ability' in t for t in test_types):
                technical_count += 1
                category = "📘 TECHNICAL"
            elif any('Personality' in t or 'Behavior' in t for t in test_types):
                behavioral_count += 1
                category = "👤 BEHAVIORAL"
            else:
                category = "🔄 OTHER"

            print(f"\n{i}. [{category}] {assessment['name']}")
            print(f"   Types: {test_types_str}")
            print(f"   Duration: {assessment['duration']} min")
            if len(assessment.get('description', '')) > 100:
                print(f"   Description: {assessment['description'][:100]}...")
            else:
                print(f"   Description: {assessment['description']}")

        print(f"\n📊 Balance Analysis:")
        print(f"   Technical: {technical_count}/10 ({technical_count*10}%)")
        print(f"   Behavioral: {behavioral_count}/10 ({behavioral_count*10}%)")
        print(f"   Other: {10-technical_count-behavioral_count}/10")

        # Check if balance makes sense
        if 'communication' in query.lower() or 'collaborate' in query.lower() or 'leadership' in query.lower():
            print(f"\n⚠️  Query suggests balance needed (technical + soft skills)")
            if behavioral_count < 3:
                print(f"   ❌ Insufficient behavioral assessments ({behavioral_count}/10)")
            else:
                print(f"   ✅ Good balance")

if __name__ == "__main__":
    test_queries()
