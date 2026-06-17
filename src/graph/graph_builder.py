from src.graph.neo4j_client import neo4j_manager


richest_people_query = """
LOAD CSV WITH HEADERS FROM
"https://raw.githubusercontent.com/Britto1221/adaptive-graphrag/main/data/public/top_30_richest_people_company_details.csv"
AS row

WITH row
WHERE trim(coalesce(row.name, '')) <> ''

// Create the person
MERGE (p:Person {name: trim(row.name)})

SET p.personId =
        CASE
            WHEN trim(coalesce(row.personId, '')) = ''
            THEN null
            ELSE toInteger(row.personId)
        END,

    p.rank =
        CASE
            WHEN trim(coalesce(row.rank, '')) = ''
            THEN null
            ELSE toInteger(row.rank)
        END,

    p.birthYear =
        CASE
            WHEN trim(coalesce(row.birthYear, '')) = ''
            THEN null
            ELSE toInteger(row.birthYear)
        END,

    p.country = trim(row.country),

    p.estimatedNetWorthUSDBillions =
        CASE
            WHEN trim(
                coalesce(
                    row.estimatedNetWorthUSDBillions,
                    ''
                )
            ) = ''
            THEN null
            ELSE toFloat(
                row.estimatedNetWorthUSDBillions
            )
        END,

    p.sourceOfWealth = trim(row.sourceOfWealth),

    p.snapshotDate =
        CASE
            WHEN trim(coalesce(row.snapshotDate, '')) = ''
            THEN null
            ELSE date(row.snapshotDate)
        END,

    p.description = row.description,

    p.relationshipTypes = [
        relationshipType IN split(
            coalesce(row.relationshipTypes, ''),
            '|'
        )
        WHERE trim(relationshipType) <> ''
        |
        trim(relationshipType)
    ],

    p.sourceUrls = [
        sourceUrl IN split(
            coalesce(row.sourceUrls, ''),
            '|'
        )
        WHERE trim(sourceUrl) <> ''
        |
        trim(sourceUrl)
    ]


// Create companies
FOREACH (
    companyName IN [
        value IN split(
            coalesce(row.companies, ''),
            '|'
        )
        WHERE trim(value) <> ''
        |
        trim(value)
    ]
    |

    MERGE (company:Company {
        name: companyName
    })

    MERGE (p)-[
        relationship:ASSOCIATED_WITH
    ]->(company)

    SET relationship.sourceOfWealth =
            row.sourceOfWealth,
        relationship.snapshotDate =
            row.snapshotDate
)


// Create roles
FOREACH (
    roleName IN [
        value IN split(
            coalesce(row.roles, ''),
            '|'
        )
        WHERE trim(value) <> ''
        |
        trim(value)
    ]
    |

    MERGE (role:Role {
        name: roleName
    })

    MERGE (p)-[:HAS_ROLE]->(role)
)


// Create industries
FOREACH (
    industryName IN [
        value IN split(
            coalesce(row.industries, ''),
            '|'
        )
        WHERE trim(value) <> ''
        |
        trim(value)
    ]
    |

    MERGE (industry:Industry {
        name: industryName
    })

    MERGE (p)-[
        :HAS_BUSINESS_INTEREST_IN
    ]->(industry)
)


// Create connections between wealthy people
FOREACH (
    relatedPersonName IN [
        value IN split(
            coalesce(row.relatedPeople, ''),
            '|'
        )
        WHERE trim(value) <> ''
        |
        trim(value)
    ]
    |

    MERGE (relatedPerson:Person {
        name: relatedPersonName
    })

    MERGE (p)-[
        relationship:BUSINESS_RELATED_TO
    ]->(relatedPerson)

    SET relationship.snapshotDate =
        row.snapshotDate
)


// Store natural-language company relationship facts
FOREACH (
    factText IN [
        value IN split(
            coalesce(
                row.companyRelationships,
                ''
            ),
            '|'
        )
        WHERE trim(value) <> ''
        |
        trim(value)
    ]
    |

    MERGE (fact:RelationshipFact {
        description: factText
    })

    SET fact.snapshotDate =
            row.snapshotDate,
        fact.sourceUrls = [
            sourceUrl IN split(
                coalesce(row.sourceUrls, ''),
                '|'
            )
            WHERE trim(sourceUrl) <> ''
            |
            trim(sourceUrl)
        ]

    MERGE (p)-[
        :HAS_RELATIONSHIP_FACT
    ]->(fact)
)
"""


def build_graph() -> None:
    graph = neo4j_manager()

    graph.query(
        """
        CREATE CONSTRAINT person_name_unique
        IF NOT EXISTS
        FOR (person:Person)
        REQUIRE person.name IS UNIQUE
        """
    )

    graph.query(
        """
        CREATE CONSTRAINT company_name_unique
        IF NOT EXISTS
        FOR (company:Company)
        REQUIRE company.name IS UNIQUE
        """
    )

    graph.query(
        """
        CREATE CONSTRAINT role_name_unique
        IF NOT EXISTS
        FOR (role:Role)
        REQUIRE role.name IS UNIQUE
        """
    )

    graph.query(
        """
        CREATE CONSTRAINT industry_name_unique
        IF NOT EXISTS
        FOR (industry:Industry)
        REQUIRE industry.name IS UNIQUE
        """
    )

    graph.query(richest_people_query)

    graph.refresh_schema()

    print("Richest-people graph imported successfully.")
    print(graph.schema)


if __name__ == "__main__":
    build_graph()