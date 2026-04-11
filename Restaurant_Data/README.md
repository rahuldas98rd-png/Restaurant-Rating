# 🍽️ Restaurant Dataset

A comprehensive global restaurant dataset, containing details on **9,551 restaurants** across multiple countries. The dataset covers restaurant identity, geographic location, cuisine types, pricing, service options, and customer ratings — making it well-suited for exploratory data analysis, recommendation systems, geospatial analysis, and hospitality research.

---

## 📁 File Information

| Property | Details |
|---|---|
| **File name** | `Dataset.csv` |
| **Format** | CSV (Comma-Separated Values) |
| **Encoding** | UTF-8 with BOM |
| **Total rows** | 9,551 (excluding header) |
| **Total columns** | 21 |
| **Schema file** | `schema.yaml` |

---

## 📊 Dataset Overview

Each row represents a single restaurant. The dataset spans multiple countries and cities, capturing a snapshot of restaurant listings from the Zomato platform. It includes both quantitative fields (coordinates, cost, ratings, votes) and categorical fields (cuisine type, service availability, price tier).

---

## 🗂️ Column Reference

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `Restaurant ID` | Integer | Unique identifier for each restaurant |
| 2 | `Restaurant Name` | String | Name of the restaurant |
| 3 | `Country Code` | Integer | Numeric code representing the restaurant's country |
| 4 | `City` | String | City where the restaurant is located |
| 5 | `Address` | String | Full street address |
| 6 | `Locality` | String | Neighborhood or locality within the city |
| 7 | `Locality Verbose` | String | Full locality name with city context |
| 8 | `Longitude` | Float | Geographic longitude coordinate |
| 9 | `Latitude` | Float | Geographic latitude coordinate |
| 10 | `Cuisines` | String | Comma-separated list of cuisines offered |
| 11 | `Average Cost for two` | Integer | Estimated cost for two people (in local currency) |
| 12 | `Currency` | String | Local currency name and symbol |
| 13 | `Has Table booking` | String | Whether table reservations are accepted (`Yes`/`No`) |
| 14 | `Has Online delivery` | String | Whether online delivery is available (`Yes`/`No`) |
| 15 | `Is delivering now` | String | Whether the restaurant is currently delivering (`Yes`/`No`) |
| 16 | `Switch to order menu` | String | Whether the order menu switch is available (`Yes`/`No`) |
| 17 | `Price range` | Integer | Price tier from `1` (budget) to `4` (premium) |
| 18 | `Aggregate rating` | Float | Customer rating on a scale of `0.0` to `5.0` |
| 19 | `Rating color` | String | Color label associated with the rating band |
| 20 | `Rating text` | String | Textual label for the rating band |
| 21 | `Votes` | Integer | Total number of customer votes/reviews |

---

## ⭐ Rating Scale

| Rating Color | Rating Text | Score Range |
|---|---|---|
| 🟢 Dark Green | Excellent | 4.5 – 5.0 |
| 🟩 Green | Very Good | 4.0 – 4.4 |
| 🟡 Yellow | Good | 3.5 – 3.9 |
| 🟠 Orange | Average | 3.0 – 3.4 |
| 🔴 Red | Poor | 2.5 – 2.9 |
| ⚪ White | Not rated | 0.0 |

---

## 💲 Price Range

| Value | Tier |
|---|---|
| `1` | Budget |
| `2` | Moderate |
| `3` | Expensive |
| `4` | Premium |

---

## 🌍 Sample Data

| Restaurant ID | Restaurant Name | City | Cuisines | Avg Cost (2) | Aggregate Rating | Rating Text |
|---|---|---|---|---|---|---|
| 6317637 | Le Petit Souffle | Makati City | French, Japanese, Desserts | 1100 | 4.8 | Excellent |
| 6304287 | Izakaya Kikufuji | Makati City | Japanese | 1200 | 4.5 | Excellent |
| 6300002 | Heat - Edsa Shangri-La | Mandaluyong City | Seafood, Asian, Filipino, Indian | 4000 | 4.4 | Very Good |
| 5946519 | Starbucks | Istanbul | Cafe | 30 | 4.9 | Excellent |

---

## 📝 Notes

- **Currency varies by country.** The `Average Cost for two` field must always be interpreted alongside the `Currency` column, as values are not normalized to a single currency.
- **Multi-cuisine entries.** The `Cuisines` field may contain multiple comma-separated values for restaurants offering diverse menus.
- **Zero ratings.** An `Aggregate rating` of `0.0` indicates the restaurant has not yet been rated (corresponding `Rating text` will be `Not rated`).
- **Encoding.** The file uses a UTF-8 BOM encoding. Some parsers may need explicit BOM handling (e.g., `encoding='utf-8-sig'` in Python's pandas).
- **Coordinate precision.** Longitude and Latitude values vary in decimal precision across records.

---
