USE air_quality_db;


-- ============================================================
-- 1. Average PM2.5 by City
-- ============================================================

SELECT
    city,
    ROUND(AVG(pm25), 2) AS avg_pm25
FROM air_quality
GROUP BY city
ORDER BY avg_pm25 DESC;


-- ============================================================
-- 2. Monthly Average PM2.5
-- ============================================================

SELECT
    DATE_FORMAT(recorded_at, '%%Y-%%m') AS month,
    ROUND(AVG(pm25), 2) AS avg_pm25
FROM air_quality
GROUP BY month
ORDER BY month;


-- ============================================================
-- 3. Peak Pollution Hours
-- ============================================================

SELECT
    HOUR(recorded_at) AS hour,
    ROUND(AVG(pm25), 2) AS avg_pm25
FROM air_quality
GROUP BY HOUR(recorded_at)
ORDER BY avg_pm25 DESC;


-- ============================================================
-- 4. Rank Cities by Average PM2.5
-- ============================================================

SELECT
    city,
    ROUND(AVG(pm25), 2) AS avg_pm25,
    RANK() OVER (
        ORDER BY AVG(pm25) DESC
    ) AS pollution_rank
FROM air_quality
GROUP BY city
ORDER BY pollution_rank;


-- ============================================================
-- 5. Monthly PM2.5 Change Using LAG()
-- ============================================================

WITH monthly_pm25 AS (

    SELECT
        DATE_FORMAT(recorded_at, '%Y-%m') AS month,
        AVG(pm25) AS avg_pm25
    FROM air_quality
    GROUP BY DATE_FORMAT(recorded_at, '%Y-%m')

)

SELECT
    month,
    ROUND(avg_pm25, 2) AS avg_pm25,

    ROUND(
        avg_pm25
        - LAG(avg_pm25) OVER (
            ORDER BY month
        ),
        2
    ) AS change_from_previous_month

FROM monthly_pm25
ORDER BY month;


-- ============================================================
-- 6. Top 10 Highest PM2.5 Observations
-- ============================================================

SELECT
    city,
    recorded_at,
    pm25,
    pm10,
    no2,
    nox
FROM air_quality
ORDER BY pm25 DESC
LIMIT 10;


-- ============================================================
-- 7. City-Level PM2.5 vs PM10
-- ============================================================

SELECT
    city,
    ROUND(AVG(pm25), 2) AS avg_pm25,
    ROUND(AVG(pm10), 2) AS avg_pm10
FROM air_quality
GROUP BY city
ORDER BY avg_pm25 DESC;


-- ============================================================
-- 8. Pollution by Weekend vs Weekday
-- ============================================================

SELECT
    CASE
        WHEN DAYOFWEEK(recorded_at) IN (1, 7)
        THEN 'Weekend'
        ELSE 'Weekday'
    END AS day_type,

    ROUND(AVG(pm25), 2) AS avg_pm25

FROM air_quality

GROUP BY
    CASE
        WHEN DAYOFWEEK(recorded_at) IN (1, 7)
        THEN 'Weekend'
        ELSE 'Weekday'
    END;