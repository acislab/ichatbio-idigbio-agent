- If the user places a search term in quotes, always use the full text that is quoted, do not break it up.

- If the user specifies an unquoted binomial species name (e.g. "Homo sapiens", without additional authorship, variation, subspecies, etc.
information), break it up using the "genus" and "specificepithet" fields instead of "scientificname".
 
- If the user specifies that they want to search for an exact scientific name, use the "scientificname" field and set "taxonrank" if known. This is especially important at the species level.

- The "scientificname" field is for all taxon ranks. If you want to only match species names, for example, you MUST also set "taxonrank" to "species".

- Searching by collector name is unlikely to retrieve all records collected by an individual. It can help to search using many different variations of the name, for example using the first initial, with or without a middle name or initial, placing the last name first, using shortened or full names (e.g., someone may go by both Pam and Pamela), etc.
 
- Searching by lists performs an OR operation. For example, a search for "genus":["Ursus","Puffinus"] will return Ursus
records and ALSO Puffinus records, it will NOT return co-occurrences of Ursus and Puffinus.

- Do not choose search parameters that only partially fulfill the user's request. Instead, you should abort (don't set any search parameters) and explain why.

- Only use the "limit" and "count" fields if explicitly asks to limit the amount of information retrieved. For example, if the user wants to know who collected the most records of a species, it's ok to retrieve the full list of record counts per collector.