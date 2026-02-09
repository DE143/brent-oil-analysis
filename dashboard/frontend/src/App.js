import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar } from 'recharts';
import { DateRangePicker } from 'react-date-range';
import 'react-date-range/dist/styles.css';
import 'react-date-range/dist/theme/default.css';
import './App.css';

function App() {
  const [priceData, setPriceData] = useState([]);
  const [events, setEvents] = useState([]);
  const [changePoints, setChangePoints] = useState([]);
  const [correlations, setCorrelations] = useState([]);
  const [summary, setSummary] = useState({});
  const [dateRange, setDateRange] = useState([
    {
      startDate: new Date('1987-05-20'),
      endDate: new Date('2022-09-30'),
      key: 'selection'
    }
  ]);
  const [selectedEvent, setSelectedEvent] = useState(null);

  useEffect(() => {
    fetchData();
  }, [dateRange]);

  const fetchData = async () => {
    const start = dateRange[0].startDate.toISOString().split('T')[0];
    const end = dateRange[0].endDate.toISOString().split('T')[0];

    // Fetch price data
    const priceRes = await fetch(`http://localhost:5000/api/prices?start=${start}&end=${end}`);
    const priceData = await priceRes.json();
    setPriceData(priceData);

    // Fetch events
    const eventsRes = await fetch(`http://localhost:5000/api/events?start=${start}&end=${end}`);
    const eventsData = await eventsRes.json();
    setEvents(eventsData);

    // Fetch change points
    const cpRes = await fetch('http://localhost:5000/api/change-points');
    const cpData = await cpRes.json();
    setChangePoints(cpData);

    // Fetch correlations
    const corrRes = await fetch('http://localhost:5000/api/correlations');
    const corrData = await corrRes.json();
    setCorrelations(corrData);

    // Fetch summary
    const summaryRes = await fetch('http://localhost:5000/api/summary-stats');
    const summaryData = await summaryRes.json();
    setSummary(summaryData);
  };

  const handleEventSelect = (event) => {
    setSelectedEvent(event);
  };

  const formatCurrency = (value) => {
    return `$${value.toFixed(2)}`;
  };

  const formatPercent = (value) => {
    return `${value.toFixed(2)}%`;
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Brent Oil Price Analysis Dashboard</h1>
        <p>Birhan Energies - Data-Driven Energy Insights</p>
      </header>

      <div className="controls">
        <DateRangePicker
          ranges={dateRange}
          onChange={item => setDateRange([item.selection])}
          minDate={new Date('1987-05-20')}
          maxDate={new Date('2022-09-30')}
        />
      </div>

      <div className="summary-cards">
        <div className="card">
          <h3>Average Price</h3>
          <p className="value">{formatCurrency(summary.average_price || 0)}</p>
        </div>
        <div className="card">
          <h3>Total Events</h3>
          <p className="value">{summary.total_events || 0}</p>
        </div>
        <div className="card">
          <h3>Change Points</h3>
          <p className="value">{summary.detected_change_points || 0}</p>
        </div>
        <div className="card">
          <h3>Avg Volatility</h3>
          <p className="value">{formatPercent((summary.average_volatility || 0) * 100)}</p>
        </div>
      </div>

      <div className="charts-container">
        <div className="chart-row">
          <div className="chart-wrapper">
            <h3>Price History with Events</h3>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={priceData.dates?.map((date, i) => ({
                date: new Date(date).toLocaleDateString(),
                price: priceData.prices?.[i],
                return: priceData.returns?.[i]
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis label={{ value: 'Price (USD)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [formatCurrency(value), 'Price']} />
                <Legend />
                <Line type="monotone" dataKey="price" stroke="#8884d8" strokeWidth={2} dot={false} name="Brent Price" />
                {events.map((event, i) => (
                  <Line
                    key={i}
                    data={[{ date: new Date(event.Date).toLocaleDateString(), event: event['Event Name'] }]}
                    dataKey="event"
                    stroke="#ff7300"
                    dot={{ r: 6 }}
                    activeDot={{ r: 8 }}
                    name={event['Event Name']}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-row">
          <div className="chart-wrapper">
            <h3>Change Point Impacts</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={changePoints}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis label={{ value: 'Price (USD)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [formatCurrency(value), 'Price']} />
                <Legend />
                <Bar dataKey="mean_before" fill="#8884d8" name="Before Change" />
                <Bar dataKey="mean_after" fill="#82ca9d" name="After Change" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-wrapper">
            <h3>Event Impact Correlations</h3>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="price_change_percent" label={{ value: 'Price Change (%)', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Days Difference', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value, name) => [name === 'price_change_percent' ? formatPercent(value) : value, name]} />
                <Legend />
                <Scatter name="Event Impacts" data={correlations} fill="#ff7300" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="events-table">
        <h3>Key Events and Impacts</h3>
        <table>
          <thead>
            <tr>
              <th>Event</th>
              <th>Date</th>
              <th>Category</th>
              <th>Price Change</th>
              <th>Closest Change Point</th>
            </tr>
          </thead>
          <tbody>
            {correlations.slice(0, 10).map((corr, i) => (
              <tr key={i} onClick={() => handleEventSelect(corr)} className={selectedEvent?.event_name === corr.event_name ? 'selected' : ''}>
                <td>{corr.event_name}</td>
                <td>{new Date(corr.event_date).toLocaleDateString()}</td>
                <td>{corr.event_category}</td>
                <td className={corr.price_change_percent > 0 ? 'positive' : 'negative'}>
                  {formatPercent(corr.price_change_percent)}
                </td>
                <td>{new Date(corr.change_point_date).toLocaleDateString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selectedEvent && (
        <div className="event-detail">
          <h3>Event Detail: {selectedEvent.event_name}</h3>
          <p><strong>Date:</strong> {new Date(selectedEvent.event_date).toLocaleDateString()}</p>
          <p><strong>Price Impact:</strong> <span className={selectedEvent.price_change_percent > 0 ? 'positive' : 'negative'}>
            {formatPercent(selectedEvent.price_change_percent)}
          </span></p>
          <p><strong>Pre-Event Average:</strong> {formatCurrency(selectedEvent.pre_event_mean)}</p>
          <p><strong>Post-Event Average:</strong> {formatCurrency(selectedEvent.post_event_mean)}</p>
          <p><strong>Days to Nearest Change Point:</strong> {selectedEvent.days_difference} days</p>
        </div>
      )}
    </div>
  );
}

export default App;