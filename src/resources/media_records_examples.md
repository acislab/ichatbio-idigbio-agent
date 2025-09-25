Prefer to fill out the "mq" object. Only fill out the "rq" object if the user specifies occurrence record-related
criteria such as species taxonomy or collection event details.

## Example 1

Request: "Audio of homo sapiens"

```
"mq": {
    "mediatype": "sounds"
},
"rq": {
    "genus": "Homo",
    "specificepithet": "sapiens"
}
```