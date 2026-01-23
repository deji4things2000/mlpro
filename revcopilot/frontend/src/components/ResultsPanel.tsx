import React from 'react';

const ResultsPanel = ({ results }) => {
    return (
        <div className="results-panel">
            <h2 className="text-xl font-bold">Results</h2>
            {results.length === 0 ? (
                <p>No results available.</p>
            ) : (
                <ul>
                    {results.map((result, index) => (
                        <li key={index} className="result-item">
                            <h3 className="font-semibold">{result.title}</h3>
                            <p>{result.description}</p>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
};

export default ResultsPanel;