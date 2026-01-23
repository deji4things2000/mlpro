import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import axios from 'axios';

const ResultPage = () => {
    const router = useRouter();
    const { id } = router.query;
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (id) {
            const fetchResult = async () => {
                try {
                    const response = await axios.get(`/api/results/${id}`);
                    setResult(response.data);
                } catch (err) {
                    setError(err);
                } finally {
                    setLoading(false);
                }
            };

            fetchResult();
        }
    }, [id]);

    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error loading result: {error.message}</div>;

    return (
        <div>
            <h1>Result for ID: {id}</h1>
            <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
    );
};

export default ResultPage;