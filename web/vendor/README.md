# Vendor scripts

| File | Version | License | Source |
|---|---|---|---|
| `marked.min.js` | 12.0.2 | MIT | https://www.npmjs.com/package/marked |
| `purify.min.js` | 3.1.6 | Apache-2.0 / MPL-2.0 | https://www.npmjs.com/package/dompurify |

Markdown is rendered with Marked and sanitized with DOMPurify before it is
inserted into the transcript. Model output is always treated as untrusted
content; raw HTML, images, forms, and embedded frames are stripped.
