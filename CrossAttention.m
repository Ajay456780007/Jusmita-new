function Z = CrossAttention(X, Y, numHeads, WQ, WK, WV, WO)
    queries = WQ * X;
    keys = WK * Y;
    values = WV * Y;
    A = attention(queries, keys, values, numHeads, 'DataFormat', 'CBT');
    Z = WO * A;
end