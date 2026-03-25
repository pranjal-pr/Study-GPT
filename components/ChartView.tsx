"use client";

import { motion } from "framer-motion";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { VisualPayload } from "@/lib/types";
import { inferChartConfig } from "@/lib/visuals";

const COLORS = ["#0f766e", "#f59e0b", "#2563eb", "#dc2626", "#7c3aed"];

interface ChartViewProps {
  visual: VisualPayload;
}

export function ChartView({ visual }: ChartViewProps) {
  const config = inferChartConfig(visual);
  if (!config) {
    return (
      <div className="flex h-full min-h-[22rem] items-center justify-center rounded-[1.6rem] border border-border bg-muted/35 text-sm text-muted-foreground">
        The model did not return enough structured data for a chart yet.
      </div>
    );
  }

  const { chartType, seriesKeys, xKey } = config;
  const data = visual.chartData;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="h-full min-h-[22rem] rounded-[1.6rem] border border-border bg-background/55 p-3"
    >
      <ResponsiveContainer width="100%" height="100%">
        {chartType === "pie" ? (
          <PieChart>
            <Tooltip />
            <Legend />
            <Pie
              data={data}
              dataKey={seriesKeys[0] ?? "value"}
              nameKey={xKey}
              innerRadius={62}
              outerRadius={92}
              paddingAngle={3}
              animationDuration={900}
            >
              {data.map((entry, index) => (
                <Cell
                  key={`${String(entry[xKey])}-${index}`}
                  fill={COLORS[index % COLORS.length]}
                />
              ))}
            </Pie>
          </PieChart>
        ) : chartType === "area" ? (
          <AreaChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.18)" />
            <XAxis dataKey={xKey} />
            <YAxis />
            <Tooltip />
            <Legend />
            {seriesKeys.map((key, index) => (
              <Area
                key={key}
                type="monotone"
                dataKey={key}
                stroke={COLORS[index % COLORS.length]}
                fill={COLORS[index % COLORS.length]}
                fillOpacity={0.22}
                animationDuration={900}
              />
            ))}
          </AreaChart>
        ) : chartType === "line" ? (
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.18)" />
            <XAxis dataKey={xKey} />
            <YAxis />
            <Tooltip />
            <Legend />
            {seriesKeys.map((key, index) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={COLORS[index % COLORS.length]}
                strokeWidth={3}
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
                animationDuration={900}
              />
            ))}
          </LineChart>
        ) : (
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.18)" />
            <XAxis dataKey={xKey} />
            <YAxis />
            <Tooltip />
            <Legend />
            {seriesKeys.map((key, index) => (
              <Bar
                key={key}
                dataKey={key}
                radius={[10, 10, 0, 0]}
                fill={COLORS[index % COLORS.length]}
                animationDuration={900}
              />
            ))}
          </BarChart>
        )}
      </ResponsiveContainer>
    </motion.div>
  );
}
