"""
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const data = [
  { name: 'AP (IoU 0.50-0.95)', all: 0.485, medium: 0.276, large: 0.488 },
  { name: 'AP (IoU 0.50)', all: 0.762, medium: 0.276, large: 0.488 },
  { name: 'AP (IoU 0.75)', all: 0.532, medium: 0.276, large: 0.488 },
  { name: 'AR (max 100)', all: 0.628, medium: 0.275, large: 0.631 },
];

const ResultsChart = () => (
  <ResponsiveContainer width="100%" height={300}>
    <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="name" />
      <YAxis />
      <Tooltip />
      <Legend />
      <Bar dataKey="all" fill="#8884d8" name="All Areas" />
      <Bar dataKey="medium" fill="#82ca9d" name="Medium Objects" />
      <Bar dataKey="large" fill="#ffc658" name="Large Objects" />
    </BarChart>
  </ResponsiveContainer>
);

export default ResultsChart;
"""