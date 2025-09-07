-- Vendor performance summary
SELECT 
    v.vendor_id,
    v.vendor_name,
    v.category,
    COUNT(po.po_id) AS total_orders,
    SUM(po.total_amount) AS total_spend,
    AVG(po.total_amount) AS avg_order_value
FROM vendors v
LEFT JOIN purchase_orders po ON v.vendor_id = po.vendor_id
GROUP BY v.vendor_id, v.vendor_name, v.category
ORDER BY total_spend DESC;

-- On-time delivery rate (simplified)
SELECT 
    v.vendor_id,
    v.vendor_name,
    COUNT(po.po_id) AS total_orders,
    SUM(CASE WHEN po.status = 'Delivered' THEN 1 ELSE 0 END) AS delivered_orders,
    ROUND(SUM(CASE WHEN po.status = 'Delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(po.po_id), 2) AS delivery_rate
FROM vendors v
LEFT JOIN purchase_orders po ON v.vendor_id = po.vendor_id
GROUP BY v.vendor_id, v.vendor_name
ORDER BY delivery_rate DESC;