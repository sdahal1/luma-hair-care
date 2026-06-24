# Luma Demo — AWS Resource Manifest

This file is the inventory of every AWS resource provisioned for the Luma
Knowledge Base demo. It exists so teardown is deterministic: to clean up,
delete everything listed here (or everything tagged `Project=luma-demo`).

- **Account:** 620958831114
- **Region:** us-east-2
- **Tag on every resource:** `Project=luma-demo`
- **Embedding model:** amazon.titan-embed-text-v2:0 (1024 dims)
- **Status legend:** PLANNED → CREATED (with id/ARN) → DELETED

## Resources

| # | Resource | Type | Name / ID / ARN | Status | Notes |
|---|----------|------|-----------------|--------|-------|
| 1 | Source docs bucket | S3 bucket | `arn:aws:s3:::luma-demo-kb-docs-620958831114` | CREATED | 10 docs uploaded; tagged Project=luma-demo; empty before delete |
| 2 | Vector store bucket | S3 Vectors vector bucket | `arn:aws:s3vectors:us-east-2:620958831114:bucket/luma-demo-vectors-620958831114` | CREATED | Dedicated vector bucket; teardown deletes index then bucket |
| 3 | Vector index | S3 Vectors index | `luma-demo-index` | CREATED | 1024 dims, cosine; non-filterable keys AMAZON_BEDROCK_TEXT/_METADATA |
| 4 | KB service role | IAM role | `arn:aws:iam::620958831114:role/AmazonBedrockExecutionRoleForKB-luma-demo` | CREATED | Trust bedrock.amazonaws.com; tagged Project=luma-demo |
| 5 | KB role policies | IAM inline policy | `LumaKBPermissions` (on role #4) | CREATED | Titan invoke + S3 read + s3vectors (incl CreateIndex) |
| 6 | Knowledge Base | Bedrock KB | `luma-demo-kb` / id `0LBF0TJ6HL` | CREATED | VECTOR; Titan v2 (1024 dims); S3 Vectors storage |
| 7 | Data source | Bedrock KB data source | id `PAA2RFNH78` (luma-demo-kb-s3-source) | CREATED | FIXED_SIZE 300 tokens / 20% overlap |

## Teardown order (reverse of creation)

1. Delete data source (#7)
2. Delete Knowledge Base (#6)
3. Delete IAM inline policies + role (#4, #5)
4. Delete S3 Vectors index then vector bucket (#3, #2)
5. Empty then delete source S3 bucket (#1)
6. Confirm nothing remains tagged `Project=luma-demo`

## Notes

- The IAM caller used to provision is `raabdahlCLI2` (an IAM user).
- After the KB is created, the role trust policy should be tightened from
  `knowledge-base/*` to the specific KB id (security best practice).
- Cost backstop: keep an AWS Budget + Cost Anomaly Detection alert on this account.

## Provisioning status

- Provisioned 2026-06-24 in us-east-2. All 7 resources CREATED.
- Initial ingestion COMPLETE (job `T3U4VVOE7U`): 10/10 docs indexed, 0 failed.
- Retrieval verified: the "safe for color-treated hair?" query returns the
  Define Curl Cream doc (score 0.86).
- `KNOWLEDGE_BASE_ID=0LBF0TJ6HL` written to `backend/.env`.
- Optional hardening (not yet done): tighten the role trust policy `aws:SourceArn`
  from `knowledge-base/*` to `knowledge-base/0LBF0TJ6HL`.
