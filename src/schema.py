"""
This module provides pydantic schema used to craft and validate iDigBio API queries.
"""

from datetime import date
from typing import Optional, List, Union, Literal

from pydantic import Field, BaseModel, field_validator
from pydantic_core import PydanticCustomError


class DateRange(BaseModel):
    type: Literal["range"]
    gte: Optional[date] = Field(
        None,
        description="The start date of the range",
        examples=["1900-3-14", "2024-01-01"],
    )
    lte: Optional[date] = Field(
        None,
        description="The end date of the range.",
        examples=["1900-12-20", "2024-02-01"],
    )


class Existence(BaseModel):
    type: Literal["exists", "missing"]


Date = Union[date, DateRange, Existence]
String = Union[str, List[str], Existence]
Bool = Union[bool, Existence]
Float = Union[float, Existence]
Int = Union[int, Existence]


class Coordinate(BaseModel):
    """
    Represents a geographic coordinate with latitude and longitude.
    """

    lat: float = Field(..., description="latitude")
    lon: float = Field(..., description="longitude")

    @field_validator("lat", mode="after")
    @classmethod
    def validate_latitude(cls, v):
        if v is None:
            return v
        if not (-90 <= v <= 90):
            raise PydanticCustomError(
                "geopoint_range_error",
                "Error: Invalid latitude value: {latitude} is not in range [-90, +90]",
                dict(latitude=v, terminal=True),
            )
        return v

    @field_validator("lon", mode="after")
    @classmethod
    def validate_longitude(cls, v):
        if v is None:
            return v
        if not (-180 <= v <= 180):
            raise PydanticCustomError(
                "geopoint_range_error",
                "Error: Invalid longitude value: {longitude} is not in range [-180, +180]",
                dict(longitude=v, terminal=True),
            )
        return v


class GeoPoint(BaseModel):
    """
    This schema represents a location on earth.
    Supports two types:
    - geo_distance: A point with optional distance radius
    - geo_bounding_box: A rectangular area defined by top-left and bottom-right coordinates
    """

    type: Literal["geo_distance", "geo_bounding_box"] = Field(default="geo_distance")

    # Fields for geo_distance type
    lat: Optional[float] = Field(
        None, description="latitude (used only when type is geo_distance)"
    )
    lon: Optional[float] = Field(
        None, description="longitude (used only when type is geo_distance)"
    )
    distance: Optional[str] = Field(
        None,
        description="distance in kilometers with km at the end. Example: 575km (used only when type is geo_distance)",
    )

    # Fields for geo_bounding_box type
    top_left: Optional[Coordinate] = Field(
        None,
        description="Top-left coordinate of bounding box (used only when type is geo_bounding_box)",
    )
    bottom_right: Optional[Coordinate] = Field(
        None,
        description="Bottom-right coordinate of bounding box (used only when type is geo_bounding_box)",
    )

    @field_validator("lat", mode="after")
    @classmethod
    def validate_latitude(cls, v, info):
        if v is None:
            return v
        if not (-90 <= v <= 90):
            raise PydanticCustomError(
                "geopoint_range_error",
                "Error: Invalid latitude value: {latitude} is not in range [-90, +90]",
                dict(latitude=v, terminal=True),
            )
        return v

    @field_validator("lon", mode="after")
    @classmethod
    def validate_longitude(cls, v, info):
        if v is None:
            return v
        if not (-180 <= v <= 180):
            raise PydanticCustomError(
                "geopoint_range_error",
                "Error: Invalid longitude value: {longitude} is not in range [-180, +180]",
                dict(longitude=v, terminal=True),
            )
        return v

    @field_validator("top_left", "bottom_right", mode="after")
    @classmethod
    def validate_coordinates_for_type(cls, v, info):
        # Make sure coordinate fields are only present when type is geo_bounding_box
        if (
            info.content.get("type") == "geo_bounding_box"
            and v is None
            and info.field_name in info.content
        ):
            raise PydanticCustomError(
                "geo_missing_field",
                "Error: {field_name} is required when type is geo_bounding_box",
                dict(field_name=info.field_name, terminal=True),
            )
        return v

    @field_validator("lat", "lon", "distance", mode="after")
    @classmethod
    def validate_fields_for_type(cls, v, info):
        # Make sure distance point fields are only present when type is geo_distance
        if (
            info.content.get("type") == "geo_distance"
            and info.field_name in ["lat", "lon"]
            and v is None
        ):
            raise PydanticCustomError(
                "geo_missing_field",
                "Error: {field_name} is required when type is geo_distance",
                dict(field_name=info.field_name, terminal=True),
            )
        return v

    # Model-level validator to check overall consistency
    @field_validator("type")
    @classmethod
    def validate_model_consistency(cls, v, info):
        if v == "geo_distance":
            # When geo_distance, top_left and bottom_right should not be present
            if (
                info.content.get("top_left") is not None
                or info.content.get("bottom_right") is not None
            ):
                raise PydanticCustomError(
                    "geo_type_mismatch",
                    "Error: top_left and bottom_right should not be present when type is geo_distance",
                    dict(terminal=True),
                )
        elif v == "geo_bounding_box":
            # When geo_bounding_box, lat, lon, and distance should not be present
            if any(
                info.content.get(field) is not None
                for field in ["lat", "lon", "distance"]
            ):
                raise PydanticCustomError(
                    "geo_type_mismatch",
                    "Error: lat, lon, and distance should not be present when type is geo_bounding_box",
                    dict(terminal=True),
                )
        return v


class IDBRecordsQuerySchema(BaseModel):
    """
    This schema represents the iDigBio Record Query Format.
    """

    associatedsequences: Optional[String] = Field(
        None,
        description="Identifiers (e.g., GenBank accession numbers or URIs) for genetic sequence data linked to the specimen or occurrence.",
    )
    barcodevalue: Optional[String] = Field(
        None,
        description="Machine-readable barcode string printed on the physical specimen label.",
    )
    basisofrecord: Optional[String] = Field(
        None,
        description="Specific nature of the data record (e.g., PreservedSpecimen, HumanObservation, MaterialSample).",
    )
    bed: Optional[String] = Field(
        None,
        description="The full name of the lithostratigraphic bed from which a material entity was collected.",
    )
    canonicalname: Optional[String] = Field(
        None,
        description="The latinized elements of a scientific name, without authorship information, etc.",
    )
    catalognumber: Optional[String] = Field(
        None,
        description="Identifier (preferably unique) for the record within its source collection or dataset.",
    )
    class_: Optional[String] = Field(
        None,
        alias="class",
        description="The taxonomic class of an organism.",
    )
    collectioncode: Optional[String] = Field(
        None,
        description="Acronym, code, or name designating the collection from which the record is derived.",
    )
    collectionid: Optional[String] = Field(
        None,
        description="Globally unique identifier (GUID/URI) for the collection housing the material.",
    )
    collectionname: Optional[String] = Field(
        None,
        description="Human-readable name of the collection that holds the record.",
    )
    collector: Optional[String] = Field(
        None,
        description="Name(s) of the person(s) or organization(s) who recorded or collected the occurrence.",
    )
    commonname: Optional[String] = Field(
        None,
        description='Common name for a specific species. Do not use for taxonomic groups like "birds" or "mammals".',
    )
    # commonnames
    continent: Optional[String] = Field(
        None,
        description="Name of the continent containing the sampling location.",
    )
    # coordinateuncertainty
    country: Optional[String] = Field(
        None,
        description="Full, accepted country name. For example “Canada” instead of the ISO code CA.",
    )
    # countrycode
    county: Optional[String] = Field(
        None,
        description="Full, unabbreviated name of the county (or equivalent) in which the location occurs.",
    )
    # data {}
    # datasetid
    datecollected: Optional[Date] = Field(
        None,
        description="Date the specimen or observation was collected (ISO 8601 formatted).",
    )
    datemodified: Optional[Date] = Field(
        None,
        description="Most recent date on which the digital record was changed (ISO 8601 formatted).",
    )
    dqs: Optional[Float] = Field(
        None,
        description="Data quality score for the record.",
    )
    # earliestageorloweststage
    # earliesteonorlowesteonothem
    # earliestepochorlowestseries
    # earliesteraorlowesterathem
    # earliestperiodorlowestsystem: Optional[String] = None
    etag: Optional[String] = Field(
        None,
        description="Entity-tag string used by iDigBio to detect record version changes.",
    )
    eventdate: Optional[Date] = Field(
        None,
        description="Date or date range during which the collecting event occurred (ISO 8601 formatted).",
    )
    family: Optional[String] = Field(
        None,
        description="Scientific name of the family in which the taxon is classified.",
    )
    fieldnumber: Optional[String] = Field(
        None,
        description="Identifier assigned in the field to the collecting event, linking field notes and specimens.",
    )
    flags: Optional[String] = Field(
        None,
        description="Semicolon-separated data-quality or processing flags applied to the record.",
    )
    # formation
    genus: Optional[String] = Field(
        None,
        description="Scientific name of the genus in which the taxon is classified.",
    )
    # geologicalcontextid
    geopoint: Optional[GeoPoint] = Field(
        None,
        description="Decimal latitude/longitude pair (usually WGS 84) for the occurrence’s center point.",
    )
    # group
    hasImage: Optional[bool] = Field(
        None, description="True if the record has one or more associated images."
    )  # All records have this field, no need to allow existence queries
    highertaxon: Optional[String] = Field(
        None,
        description="Pipe-separated list of higher taxonomic ranks above the taxon (e.g., “Animalia | Chordata | Mammalia”).",
    )
    # highestbiostratigraphiczone
    # indexData {}
    # individualcount
    infraspecificepithet: Optional[String] = Field(
        None,
        description="Lowest infraspecific epithet of the scientific name (e.g., “oxyadenia”).",
    )
    institutioncode: Optional[String] = Field(
        None,
        description="The name (or acronym) in use by the institution having custody of the object(s) or information referred to in the record.",
    )
    institutionid: Optional[String] = Field(
        None,
        description="An identifier for the institution having custody of the object(s) or information referred to in the record.",
    )
    institutionname: Optional[String] = Field(
        None,
        description="Full name of the institution that owns or manages the collection or data.",
    )
    kingdom: Optional[String] = Field(
        None,
        description="Scientific name of the kingdom in which the taxon is classified.",
    )
    # latestageorhigheststage
    # latesteonorhighesteonothem
    # latestepochorhighestseries
    # latesteraorhighesterathem
    # latestperiodorhighestsystem: Optional[String] = None
    # lithostratigraphicterms
    locality: Optional[String] = Field(
        None,
        description="Specific descriptive text of the place where the specimen was collected or observed.",
    )
    # lowestbiostratigraphiczone
    maxdepth: Optional[Float] = Field(
        None,
        description="Greater depth (metres) below the local surface at which the record was made.",
    )
    maxelevation: Optional[Float] = Field(
        None,
        description="Upper limit of elevation (metres above sea level) at the site.",
    )
    mediarecords: Optional[String] = Field(
        None,
        description="Identifiers or URLs of media (images, audio, video) associated with the record.",
    )
    # member
    mindepth: Optional[Float] = Field(
        None,
        description="Lesser depth (metres) below the local surface at which the record was made.",
    )
    minelevation: Optional[Float] = Field(
        None,
        description="Lower limit of elevation (metres above sea level) at the site.",
    )
    municipality: Optional[String] = Field(
        None,
        description="Name of the municipality or city containing the location.",
    )
    occurrenceid: Optional[String] = Field(
        None,
        description="Globally unique identifier (GUID/URI) for the occurrence itself.",
    )
    order: Optional[String] = Field(
        None,
        description="Scientific name of the order in which the taxon is classified.",
    )
    phylum: Optional[String] = Field(
        None,
        description="Scientific name of the phylum or division in which the taxon is classified.",
    )
    # query
    recordids: Optional[String] = Field(
        None,
        description="Comma-separated list of specific iDigBio record UUIDs to include in the query.",
    )
    recordnumber: Optional[String] = Field(
        None,
        description="Collector’s number assigned to the occurrence at the time of collection.",
    )
    recordset: Optional[String] = Field(
        None,
        description="Identifier for an iDigBio recordset (dataset) used to filter the query.",
    )
    scientificname: Optional[String] = Field(
        None,
        description="Full scientific name, including authorship, applied to the organism.",
    )
    # size
    specificepithet: Optional[String] = Field(
        None,
        description="Species epithet component of the scientific name.",
    )
    # startdayofyear
    stateprovince: Optional[Union[String, List[str]]] = Field(
        None,
        description="Name of the primary administrative region (state, province, region) for the location.",
    )
    taxonid: Optional[String] = Field(
        None,
        description="An identifier for the set of dwc:Taxon information. May be a global unique identifier or an identifier specific to the data set.",
    )
    taxonomicstatus: Optional[String] = Field(
        None,
        description="The status of the use of the scientific name as a label for a taxon (e.g., accepted, invalid, misapplied).",
        examples=["invalid", "misapplied", "homotypic synonym", "accepted"],
    )
    taxonrank: Optional[String] = Field(
        None,
        description="Taxonomic rank of the most specific name",
        examples=["species", "subspecies", "genus"],
    )
    typestatus: Optional[String] = Field(
        None,
        description="A list (concatenated and separated) of nomenclatural types (type status, typified scientific name, publication) applied to the subject.",
        examples=["holotype of Pinus abies | holotype of Picea abies"],
    )
    uuid: Optional[String] = Field(
        None,
        description="An internal identifier used by iDigBio to identify the record.",
    )
    verbatimeventdate: Optional[String] = Field(
        None,
        description="Original, unaltered text of the collection date as it appears on the label or notes.",
    )
    verbatimlocality: Optional[String] = Field(
        None,
        description="Original, unaltered locality description from the specimen label.",
    )
    version: Optional[Int] = Field(
        None,
        description="Integer representing the current revision number of the record in iDigBio.",
    )
    waterbody: Optional[String] = Field(
        None,
        description="Name of the water body (ocean, sea, lake, river) in which the location occurs.",
    )

    class Config:
        json_encoders = {date: date.isoformat}


class IDBMediaQuerySchema(BaseModel):
    """
    This schema represents the iDigBio Media Query Format.
    """

    accessuri: Optional[String] = None
    datemodified: Optional[Date] = Field(
        None, description='The "datemodified" field in the original media record'
    )
    # dqs: Optional[Float] = Field(None, description="Data quality score for the mediarecord. DO not use unless specified by user.")
    etag: Optional[String] = None
    # Should be a Literal, leave commented for now to prevent undefined behavior.
    # flags: Optional[String] = None
    # format: Optional[String] = Field(None, description="Image format. Do not use this field unless the user specifies a format.")
    # All records have "hasSpecimen", no need to allow existence queries
    # hasSpecimen: Optional[bool] = Field(None,
    # description="Whether the media record is associated with a specific species "
    #             "occurrence record")
    licenselogourl: Optional[String] = None
    mediatype: Optional[Literal["images", "sounds"]] = None
    # modified: Optional[String] = None # TODO: how this is different from datemodified?
    modified: Optional[Date] = Field(
        None,
        description="Last time the media record changed in iDigBio, whether the original "
        "record or iDigBio's metadata",
    )
    recordids: Optional[String] = None
    records: Optional[String] = Field(
        None, description="UUIDs for records that are associated with the media record"
    )
    recordset: Optional[String] = Field(
        None, description="The record set that the media record is a part of"
    )
    rights: Optional[String] = None
    # tag: Optional[String] = None # TODO
    # type: Optional[String] = None
    uuid: Optional[String] = Field(
        None, description="An identifier used by iDigBio to identify the mediarecord"
    )
    version: Optional[Int] = None

    # webstatement: Optional[String] = None # TODO
    # xpixels: Optional[Int] = None
    # ypixels: Optional[Int] = None

    class Config:
        json_encoders = {date: date.isoformat}


class IDigBioRecordsApiParameters(BaseModel):
    """
    This schema represents the output containing the LLM-generated iDigBio query.
    """

    rq: IDBRecordsQuerySchema = Field(
        description="Search criteria for species occurrence records in iDigBio"
    )
    limit: Optional[int] = Field(
        100, ge=1, le=5000, description="The maximum number of records to return"
    )


class IDigBioSummaryApiParameters(BaseModel):
    """
    This schema represents the output containing the LLM-generated iDigBio query.
    """

    top_fields: str = Field(
        description="The field to break down record counts by. Defaults to "
        '"scientificname". For example, if top_fields is "country", '
        "the iDigBio API will find the 10 countries with the most records "
        "matching the search parameters. Only one top field may be "
        "specified.",
    )
    count: Optional[int] = Field(
        10,
        gt=0,
        le=5000,
        description="The maximum number of unique values to report record counts for. For "
        'example, to find 10 number of species, set "count" to 10. Or to find '
        "total number of unique species, use the maximum count allowed.",
    )
    rq: Optional[IDBRecordsQuerySchema] = Field(
        None,
        description="This is the iDigBio Query format and should contain the "
        "query "
        "generated from the user's plain text input.",
    )


class IDigBioMediaApiParameters(BaseModel):
    """
    This schema represents the output containing the LLM-generated iDigBio query.
    """

    mq: Optional[IDBMediaQuerySchema] = Field(
        None, description="Search criteria for media and media records"
    )
    rq: Optional[IDBRecordsQuerySchema] = Field(
        None, description="Search criteria for species occurrence records"
    )
    limit: Optional[int] = Field(
        None, description="The maximum number of records to return"
    )
